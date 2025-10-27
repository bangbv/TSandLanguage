import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
  LlamaForCausalLM,
  LlamaTokenizer,
)
import numpy as np


class LLAMAmodel(torch.nn.Module):
  def __init__(self, config):
    super(LLAMAmodel, self).__init__()
    self.task = config.experiment.task
    self.settings = config.model.settings
    self.model_name = config.model.name
    self.tokenizer = None
    self.DEFAULT_EOS_TOKEN = "</s>"
    self.DEFAULT_BOS_TOKEN = "<s>"
    self.DEFAULT_UNK_TOKEN = "<unk>"
    self.loaded = {}
    self.debug_mode: bool = config.debug_mode

    # Embedding conversion parameters
    self.use_embeddings = getattr(config.model, 'use_embeddings', False)
    self.embedding_dim = getattr(config.model, 'embedding_dim',
                                 768)  # Default LLaMA hidden size
    self.max_sequence_length = getattr(config.model, 'max_sequence_length', 512)

  def llama_model_string(self, model_size, chat):
    print_debug(my_print, "LLAMAmodel:llama2_model_string:chat:", chat,
                self.debug_mode)
    chat = "chat-" if chat else ""
    model_string = f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"
    print_debug(my_print, "LLAMAmodel:llama_new_model_string:", model_string,
                True)
    return model_string

  def get_tokenizer(self, model_name):
    print_debug(my_print, "LLAMAmodel:get_tokenizer:model_name:", model_name,
                self.debug_mode)
    name_parts = model_name.split("-")
    print_debug(my_print, "LLAMAmodel:get_tokenizer:name_parts:", name_parts,
                self.debug_mode)
    model_size = name_parts[0]
    print_debug(my_print, "LLAMAmodel:get_tokenizer:model_size:", model_size,
                self.debug_mode)
    chat = len(name_parts) > 1
    assert model_size in ["7b", "13b", "70b"]
    print_debug(my_print, "LLAMAmodel:get_tokenizer:chat:", chat,
                self.debug_mode)
    tokenizer = LlamaTokenizer.from_pretrained(
        self.llama_model_string(model_size, chat),
        use_fast=False,
    )
    print_debug(my_print, "LLAMAmodel:get_tokenizer:", "finish init tokenizer",
                self.debug_mode)
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
      special_tokens_dict["eos_token"] = self.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
      special_tokens_dict["bos_token"] = self.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
      special_tokens_dict["unk_token"] = self.DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

  def convert_time_series_to_embeddings(self,
      input_arr,
      input_trend_arr,
      input_season_arr,
      input_resid_arr,
      description=""):
    """
    Convert time series input strings to embedding vectors.

    Args:
        input_str: Main time series string "627, 661, 739, 723, ..."
        input_trend_str: Trend component string
        input_season_str: Seasonal component string
        input_resid_str: Residual component string
        description: Optional contextual description

    Returns:
        torch.Tensor: Embedding vector of shape [batch_size, seq_len, embedding_dim]
    """
    print_debug(my_print,
                "LLAMAmodel:convert_time_series_to_embeddings:input_str", input_arr[:100], self.debug_mode)

    main_values = input_arr
    trend_values = input_trend_arr
    season_values = input_season_arr
    resid_values = input_resid_arr
    # Ensure all components have the same length
    min_length = min(len(main_values), len(trend_values), len(season_values),
                     len(resid_values))
    main_values = main_values[:min_length]
    trend_values = trend_values[:min_length]
    season_values = season_values[:min_length]
    resid_values = resid_values[:min_length]

    # Limit to max sequence length
    if len(main_values) > self.max_sequence_length:
      main_values = main_values[-self.max_sequence_length:]
      trend_values = trend_values[-self.max_sequence_length:]
      season_values = season_values[-self.max_sequence_length:]
      resid_values = resid_values[-self.max_sequence_length:]

    # Create embedding matrix: [seq_len, 4] for the 4 components
    time_series_matrix = torch.tensor([
      main_values,
      trend_values,
      season_values,
      resid_values
    ], dtype=torch.float32).T  # Shape: [seq_len, 4]

    # Project to embedding dimension using a linear transformation
    # This creates a learnable mapping from 4D time series features to embedding_dim
    projection_layer = torch.nn.Linear(4, self.embedding_dim, bias=True)
    embeddings = projection_layer(
      time_series_matrix)  # Shape: [seq_len, embedding_dim]

    # Add positional encoding
    embeddings = self._add_positional_encoding(embeddings)

    # Add batch dimension: [1, seq_len, embedding_dim]
    embeddings = embeddings.unsqueeze(0)

    print_debug(my_print,
                "LLAMAmodel:convert_time_series_to_embeddings:output_shape",
                embeddings.shape, self.debug_mode)
    return embeddings

  def _parse_time_series_string(self, ts_string):
    """Parse time series string to numerical values."""
    print_debug(my_print, "LLAMAmodel:_parse_time_series_string:ts_string",
                ts_string, self.debug_mode)
    try:
      # Split by comma and convert to float
      values = [float(x.strip()) for x in ts_string.split(',') if x.strip()]
      return values
    except (ValueError, AttributeError):
      print_debug(my_print, "LLAMAmodel:_parse_time_series_string:error",
                  ts_string, self.debug_mode)
      return [0.0]  # Return default value on error

  def _add_positional_encoding(self, embeddings):
    """Add sinusoidal positional encoding to embeddings."""
    seq_len, embedding_dim = embeddings.shape
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

    # Create sinusoidal positional encodings
    div_term = torch.exp(
      torch.arange(0, embedding_dim, 2, dtype=torch.float32) *
      -(np.log(10000.0) / embedding_dim))

    pos_encoding = torch.zeros(seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return embeddings + pos_encoding

  def get_model_and_tokenizer(self, model_name, cache_model=False):
    print_debug(my_print, "LLAMAmodel:load_model_and_tokenizer:model_name:",
                model_name, True)
    if model_name in self.loaded:
      return self.loaded[model_name]
    name_parts = model_name.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]

    tokenizer = self.get_tokenizer(model_name)

    model = LlamaForCausalLM.from_pretrained(
        self.llama_model_string(model_size, chat),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    if cache_model:
      self.loaded[model_name] = model, tokenizer
    return model, tokenizer

  def tokenize_fn(self, str, model_name):
    print_debug(my_print, "LLAMAmodel:tokenize_fn:model_name:", model_name,
                self.debug_mode)
    tokenizer = self.get_tokenizer(model_name)
    return tokenizer(str)

  def run(self,
      input_arr, input_str,
      input_trend_arr, input_trend_str,
      input_season_arr, input_season_str,
      input_resid_arr, input_resid_str,
      description, steps, config, batch_size, num_samples, temp
  ):
    model_name = config.model.model_name
    settings = config.model.settings
    if self.task == 'forecast' and self.use_embeddings:
      return self.forecast_with_embeddings(model_name,
                                           input_arr, input_str,
                                           input_trend_arr, input_trend_str,
                                           input_season_arr, input_season_str,
                                           input_resid_arr, input_resid_str,
                                           steps, settings, batch_size,
                                           num_samples, temp
                                           )
    else:
      return self.forecast_with_tokens(model_name,
                                       input_str, input_trend_str,
                                       input_season_str, input_resid_str,
                                       steps, settings, batch_size, num_samples,
                                       temp
                                       )

  def forecast_with_embeddings(self, model_name,
      input_arr, input_str,
      input_trend_arr, input_trend_str,
      input_season_arr, input_season_str,
      input_resid_arr, input_resid_str,
      steps, settings, batch_size=5, num_samples=20, temp=0.9, top_p=0.9,
      cache_model=True):
    """Forecast using embedding vectors instead of tokenized text."""
    print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:starting",
                "embedding-based forecasting", self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast:input_trend_str:",
                input_trend_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast:input_season_str:",
                input_season_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast:input_resid_str:",
                input_resid_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast:use_embeddings:",
                self.use_embeddings, self.debug_mode)
    model, tokenizer = self.get_model_and_tokenizer(model_name,
                                                    cache_model=cache_model)
    # Convert time series to embedding vectors
    embeddings = self.convert_time_series_to_embeddings(input_arr,
                                                        input_trend_arr,
                                                        input_season_arr,
                                                        input_resid_arr
                                                        )
    embeddings = embeddings.to(dtype=torch.float16, device='cuda')  # Move to GPU

    gen_strs = []
    print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:num_samples:",
                num_samples, self.debug_mode)
    print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:batch_size:",
                batch_size, self.debug_mode)

    for _ in tqdm(range(num_samples // batch_size)):
      print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:Repeat embeddings:",
                  batch_size, self.debug_mode)
      # Repeat embeddings for batch processing
      batch_embeddings = embeddings.repeat(batch_size, 1,1)  # [batch_size, seq_len, embedding_dim]
      print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:batch_embeddings shape:",
                  batch_embeddings.shape, self.debug_mode)
      # Generate predictions using the model with embedding inputs
      with torch.no_grad():
        print_debug(my_print,
                    "LLAMAmodel:_forecast_with_embeddings:torch.no_grad():",
                    "forward method", self.debug_mode)
        # Use model's forward method with embedding inputs
        outputs = model(inputs_embeds=batch_embeddings, use_cache=True)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        print_debug(my_print,
                    "LLAMAmodel:_forecast_with_embeddings:logits shape:",
                    logits.shape, self.debug_mode)
        # Sample from the distribution
        predictions = self._sample_from_embeddings(logits, steps, temp, top_p,
                                                   settings, tokenizer)
        print_debug(my_print,
                    "LLAMAmodel:_forecast_with_embeddings:predictions:",
                    predictions, self.debug_mode)

      gen_strs.extend(predictions)
      print_debug(my_print,
                  "LLAMAmodel:_forecast_with_embeddings:batch_predictions",
                  len(predictions), self.debug_mode)

    print_debug(my_print, "LLAMAmodel:_forecast_with_embeddings:total_gen_strs",
                len(gen_strs), self.debug_mode)
    return gen_strs

  def _sample_from_embeddings(self, logits, steps, temp, top_p, settings,
      tokenizer):
    """Sample predictions from logits generated by embedding inputs."""
    batch_size = logits.shape[0]
    predictions = []

    # Get the last token predictions for each sequence in batch
    last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

    # Apply temperature scaling
    last_logits = last_logits / temp

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
      ..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1,
                                                         index=sorted_indices,
                                                         src=sorted_indices_to_remove)
    last_logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = F.softmax(last_logits, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

    # Convert to strings (this is a simplified approach)
    # In practice, you'd want more sophisticated conversion from tokens to time series values
    for i in range(batch_size):
      token_id = sampled_tokens[i].item()
      # Convert token back to numerical value (simplified)
      # This would need proper implementation based on your tokenizer
      pred_str = self._convert_token_to_timeseries_value(token_id, tokenizer,
                                                         steps, settings)
      predictions.append(pred_str)

    return predictions

  def _convert_token_to_timeseries_value(self, token_id, tokenizer, steps,
      settings):
    """Convert sampled token back to time series format."""
    # Simplified conversion - in practice, you'd need proper reverse engineering
    # of your serialization format
    token_str = tokenizer.decode([token_id])

    # Generate mock prediction in the expected format
    # This is a placeholder - you'd implement proper conversion logic
    # mock_values = np.random.normal(500, 50, steps)  # Generate some mock values
    # pred_str = ', '.join([f"{int(val)}" for val in mock_values])
    pred_str = token_str
    print_debug(my_print, "LLAMAmodel:_convert_token_to_timeseries_value",
                pred_str[:50], self.debug_mode)
    return pred_str

  def forecast_with_tokens(self, model_name,
      input_str, input_trend_str, input_season_str, input_resid_str,
      steps, settings, batch_size=5, num_samples=20, temp=0.9, top_p=0.9,
      cache_model=True):
    """Original tokenization-based forecasting method."""
    print_debug(my_print, "LLAMAModelNew:forecast_with_tokens:input_trend_str:",
                input_trend_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast_with_tokens:input_season_str:",
                input_season_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast_with_tokens:input_resid_str:",
                input_resid_str, self.debug_mode)
    print_debug(my_print, "LLAMAModelNew:forecast_with_tokens:use_embeddings:",
                self.use_embeddings, self.debug_mode)
    avg_tokens_per_step = len(
        self.tokenize_fn(input_str, model_name)['input_ids']) / len(
      input_str.split(settings.time_sep))
    max_tokens = int(avg_tokens_per_step * steps)
    model, tokenizer = self.get_model_and_tokenizer(model_name,
                                                    cache_model=cache_model)
    print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:model name",
                "model name:", self.debug_mode)
    print_debug(my_print,
                "LLAMAmodel:_forecast_with_tokens:finish load_model_and_tokenizer",
                tokenizer, self.debug_mode)
    gen_strs = []
    print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:num_samples:",
                num_samples, self.debug_mode)
    print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:batch_size:",
                batch_size, self.debug_mode)

    for _ in tqdm(range(num_samples // batch_size)):
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:_", _,
                  self.debug_mode)
      batch = tokenizer(
          [input_str],
          return_tensors="pt",
      )

      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:batch first",
                  batch, self.debug_mode)
      batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:batch dict",
                  batch, self.debug_mode)
      batch = {k: v.cuda() for k, v in batch.items()}
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:batch third",
                  batch, self.debug_mode)
      num_input_ids = batch['input_ids'].shape[1]
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:num_input_ids",
                  num_input_ids, self.debug_mode)
      good_tokens_str = list("0123456789" + settings.time_sep)
      good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in
                     good_tokens_str]
      bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:batch input",
                  batch, self.debug_mode)
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:max_tokens",
                  max_tokens, self.debug_mode)
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:temp", temp,
                  self.debug_mode)
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:top_p", top_p,
                  self.debug_mode)
      array_bad_tokens = [[t] for t in bad_tokens]
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:bad_words_ids",
                  array_bad_tokens[0:3], self.debug_mode)

      generate_ids = model.generate(
          **batch,
          do_sample=True,
          max_new_tokens=max_tokens,
          temperature=temp,
          top_p=top_p,
          bad_words_ids=[[t] for t in bad_tokens],
          renormalize_logits=True,
      )

      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:generate_ids",
                  generate_ids, self.debug_mode)
      gen_strs += tokenizer.batch_decode(
          generate_ids[:, num_input_ids:],
          skip_special_tokens=True,
          clean_up_tokenization_spaces=False
      )
      print_debug(my_print, "LLAMAmodel:_forecast_with_tokens:gen_strs",
                  gen_strs, self.debug_mode)
    return gen_strs


def print_debug(f, header, value, debug_mode=False):
  if debug_mode: f(header, value)


def my_print(header, value):
  print(f"{header}: {value}")
