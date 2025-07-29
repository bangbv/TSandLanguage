import torch
from tqdm import tqdm
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

class LLAMAmodel(torch.nn.Module):
    def __init__(self, config):
        super(LLAMAmodel, self).__init__()
        self.task = config.experiment.task
        self.settings  = config.model.settings
        self.model_name = config.model.name
        self.tokenizer = None
        self.DEFAULT_EOS_TOKEN = "</s>"
        self.DEFAULT_BOS_TOKEN = "<s>"
        self.DEFAULT_UNK_TOKEN = "<unk>"
        self.loaded = {}
        self.debug_mode: bool = config.debug_mode

    def llama_model_string(self, model_size, chat):
        print_debug(my_print, "LLAMAmodel:llama2_model_string:chat:", chat, self.debug_mode)
        chat = "chat-" if chat else ""
        model_string = f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"
        return model_string

    def load_tokenizer(self, model_name):
        print_debug(my_print, "LLAMAmodel:get_tokenizer:model_name:",model_name, self.debug_mode)
        name_parts = model_name.split("-")
        model_size = name_parts[0]
        chat = len(name_parts) > 1
        assert model_size in ["7b", "13b", "70b"]

        tokenizer = LlamaTokenizer.from_pretrained(
            self.llama_model_string(model_size, chat),
            use_fast=False,
        )

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

    def load_model_and_tokenizer(self, model_name, cache_model=False):
        print_debug(my_print, "LLAMAmodel:load_model_and_tokenizer:model_name:", model_name, True)
        if model_name in self.loaded:
            return self.loaded[model_name]
        name_parts = model_name.split("-")
        model_size = name_parts[0]
        chat = len(name_parts) > 1

        assert model_size in ["7b", "13b", "70b"]

        tokenizer = self.load_tokenizer(model_name)

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
        print_debug(my_print, "LLAMAmodel:tokenize_fn:model_name:", model_name, self.debug_mode)
        tokenizer = self.load_tokenizer(model_name)
        return tokenizer(str)

    def run(self, input_str, description, steps, config, batch_size, num_samples, temp):
        model_name = config.model.model_name
        settings = config.model.settings
        if self.task == 'forecast':
            return self.forecast(model_name, input_str, steps, settings, batch_size, num_samples, temp)

    def forecast(self, model_name, input_str, steps, settings, batch_size=5, num_samples=20, temp=0.9, top_p=0.9, cache_model=True):
        debug_mode = self.debug_mode
        avg_tokens_per_step = len(self.tokenize_fn(input_str, model_name)['input_ids']) / len(input_str.split(settings.time_sep))
        max_tokens = int(avg_tokens_per_step * steps)
        model, tokenizer = self.load_model_and_tokenizer(model_name, cache_model=cache_model)

        print_debug(my_print, "LLAMAmodel:forecast", "finish load_model_and_tokenizer:",self.debug_mode)
        gen_strs = []
        for _ in tqdm(range(num_samples // batch_size)):
            print_debug(my_print, "LLAMAmodel:forecast: batch_size", batch_size,debug_mode)
            batch = tokenizer(
                [input_str],
                return_tensors="pt",
            )

            print_debug(my_print, "LLAMAmodel:forecast: batch",batch, debug_mode)
            batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
            batch = {k: v.cuda() for k, v in batch.items()}
            num_input_ids = batch['input_ids'].shape[1]

            good_tokens_str = list("0123456789" + settings.time_sep)
            good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
            # good_tokens += [tokenizer.eos_token_id]
            bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

            # Example: encode a retrieved passage with Sentence-BERT
            # retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            # doc_text = "Vietnamese e-wallet MoMo offers pay-later loans with dynamic credit limits."
            # ext_vec = retriever.encode(doc_text, convert_to_tensor=True,device=device)  # [768]

            # proj = nn.Linear(ext_vec.shape[-1], hidden_size, bias=False).to(device).half()
            # prefix_embed = proj(ext_vec).unsqueeze(0).unsqueeze(1)  # [1, 1, 4096]

            # prompt = "Summarise the product in one sentence."
            # tok = tokenizer(prompt, return_tensors="pt").to(device)
            # tok_embeds = model.model.embed_tokens(tok.input_ids)  # [1, S, 4096]

            # inputs_embeds = torch.cat([prefix_embed, tok_embeds],dim=1)  # [1, 1+S, 4096]

            # attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,device=device)

            # generate_ids = model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attn_mask,
            #     max_new_tokens=60,
            #     do_sample=True,
            #     temperature=temp
            # )

            generate_ids = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                bad_words_ids=[[t] for t in bad_tokens],
                renormalize_logits=True,
            )

            print_debug(my_print, "LLAMAmodel:forecast: generate_ids", generate_ids,debug_mode)
            gen_strs += tokenizer.batch_decode(
                generate_ids[:, num_input_ids:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            print_debug(my_print, "LLAMAmodel:forecast: gen_strs", gen_strs, debug_mode)
        return gen_strs

def print_debug(f, header, value, debug_mode = False):
    if debug_mode : f(header, value)

def my_print(header, value):
    print(f"{header}: {value}")