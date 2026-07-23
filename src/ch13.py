# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="DXRiRkWQeDWP"
# # 第13章

# %% colab={"base_uri": "https://localhost:8080/"} id="MTvs6zwq_zMm" outputId="dfaa7ab6-4bcf-4843-f4e1-52e5644a6f53"
# !pip install dowhy==0.11
# !pip install transformers==4.38.2
# !pip install accelerate==0.28.0
# !pip install pandas==2.0.3
# !pip install numpy==1.25.2
# !pip install pyro-ppl==1.9.0

# %% [markdown] id="C1MNO1jYSFMD"
# ## リスト13.1

# %% colab={"base_uri": "https://localhost:8080/"} id="3Fs95nOq_wu4" outputId="ee34e062-f0b4-49b2-c654-b61a90965e27"
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
from dowhy.datasets import linear_dataset

n_points = 1000
data = pd.DataFrame({
    "S": np.random.binomial(n=1, p=0.5, size=n_points),
    "LC": np.random.binomial(n=1, p=0.5, size=n_points),
    "Price": np.random.normal(loc=5, scale=1, size=n_points),
})

model=CausalModel(
        data = data,
        treatment='S',
        outcome='LC',
        common_causes=['G', 'A', 'E', 'O'],
        instruments=['Price']
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand,
                                 method_name="iv.instrumental_variable",
                                 method_params={'iv_instrument_name': 'Price'})

print(estimate)

# %% [markdown] id="f9-KjHaGKcuK"
# ## リスト13.2

# %% id="QW2fSnevJSfB" colab={"base_uri": "https://localhost:8080/"} outputId="0eada49c-21db-4181-e0c2-40c0db63a6c9"
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.tokenize("Can LLMs reason counterfactually?")
print(tokens)

# %% [markdown] id="n7DQiiVaKjAK"
# ## リスト13.3

# %% id="jUwQdYq2K1YP" colab={"base_uri": "https://localhost:8080/"} outputId="c39b611c-013c-4a54-bec5-b1510d1e46fb"
input_ids = tokenizer.encode(
    "Can LLMs reason counterfactually?",
    return_tensors='pt'
)
print(input_ids)

# %% [markdown] id="2so7_RmHK-RT"
# ## リスト13.4

# %% id="hNS1GYarLANK" colab={"base_uri": "https://localhost:8080/"} outputId="0866f629-e110-4baa-ce7f-0d1e1e359c83"
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.eval()

input_text = "Can LLMs reason counterfactually?<|endoftext|>"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
for idx, token in enumerate(input_ids[0]):
    token_log_prob = log_probs[0][idx][token].item()
    print(f"Token: {tokenizer.decode(token)}" +
           f" | Log Probability: {token_log_prob}")

# %% [markdown]
# > **訳者補足**: 本文 p.506 のコードは2つ目の文字列に f が抜けており、そのままでは
# > 対数確率の数値が表示されません。ここでは f を補って実行しています。


# %% [markdown] id="hfcIfZ7gLJ4Y"
# ## リスト13.5

# %% id="h472ylfqLbVe" colab={"base_uri": "https://localhost:8080/"} outputId="81f5f6ef-a396-4845-f6ff-9623bed4ee04"
prompt = "Counterfactual reasoning would enable AI to"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=25,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(
    output[0], skip_special_tokens=True)
print(generated_text)

# %% [markdown] id="iyLSsxD2LTF9"
# ## リスト13.6

# %% id="uYq9PeUHLoUq" colab={"base_uri": "https://localhost:8080/"} outputId="9c47ae0f-b814-426a-c8dc-9eb381ec16af"
import pandas as pd
url = ("https://raw.githubusercontent.com/altdeep/"
       "causalAI/master/book/chapter%2013/"
       "king-prince-kingdom-updated.csv")
df = pd.read_csv(url)
print(df.shape[0])

print(df["King"][0] + "\n")
print(df["King"][1] + "\n")
print(df["King"][2])

print("----")
print(df["Prince"][0] + "\n")
print(df["Prince"][1] + "\n")
print(df["Prince"][2])

print("----")
print(df["Kingdom"][0] + "\n")
print(df["Kingdom"][1] + "\n")
print(df["Kingdom"][2])


# %% [markdown] id="vebDMQAdLvzw"
# ## リスト13.7

# %% id="WpEkPpNhLvB6" colab={"base_uri": "https://localhost:8080/"} outputId="085ade51-979d-482d-ed94-6a96417ec06e"
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, DataCollatorForLanguageModeling,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    Trainer, TrainingArguments)
url = ("https://raw.githubusercontent.com/altdeep/"
       "causalAI/master/book/chapter%2013/"
       "king-prince-kingdom-updated.csv")
df = pd.read_csv(url)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_phrases(phrases, max_length=40):
    return tokenizer(
        phrases,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )


# %% [markdown] id="hprEwsTxL6Gw"
# ## リスト13.8

# %% id="x7DrE9EvMEL0"
class ModelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings.input_ids)
def create_king_dataset(input_phrases):
    king_phrases = input_phrases.tolist()
    king_encodings = tokenize_phrases(king_phrases)
    king_dataset = ModelDataset(
        king_encodings, king_encodings['input_ids'])
    return king_dataset


# %% [markdown] id="yJxNyr6rMK0a"
# ## リスト13.9

# %% id="5D0Lp-VzMQtw"
def create_seq2seq_datasets(input_phrases, target_phrases):
    input_phrases_list = input_phrases.tolist()
    target_phrases_list = target_phrases.tolist()
    spit = train_test_split(
        input_phrases_list,
        target_phrases_list,
        test_size=0.1
    )
    train_inputs, val_inputs, train_targets, val_targets = spit
    train_input_encodings = tokenize_phrases(train_inputs)
    val_input_encodings = tokenize_phrases(val_inputs)
    train_target_encodings = tokenize_phrases(train_targets)
    val_target_encodings = tokenize_phrases(val_targets)
    train_dataset = ModelDataset(
        train_input_encodings, train_target_encodings['input_ids']
    )
    val_dataset = ModelDataset(
        val_input_encodings, val_target_encodings['input_ids']
    )
    return train_dataset, val_dataset


# %% [markdown] id="7H0P-cV5MVl3"
# ## リスト13.10

# %% id="eyy490usMbtx"
def train_king_model(output_dir, train_dataset,
                     model_name="gpt2-medium", epochs=4):
    king_model = AutoModelForCausalLM.from_pretrained(model_name)
    training_args_king = TrainingArguments(
      output_dir=output_dir,
      per_device_train_batch_size=32,
      overwrite_output_dir=True,
      num_train_epochs=epochs,
      save_total_limit=1,
      save_steps=len(train_dataset) // 16,
      max_grad_norm=1.0
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    trainer_king = Trainer(
        model=king_model,
        args=training_args_king,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer_king.train()
    king_model.save_pretrained(output_dir)
    return king_model


# %% [markdown] id="nrIZJ0aWMgiL"
# ## リスト13.11

# %% id="nBT2Dk5gMpu5"
def train_seq2seq_model(output_dir, train_dataset, val_dataset,
                        model_name="facebook/bart-base",
                        epochs=4):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        save_steps=len(train_dataset) // 16,
        learning_rate=3e-5,
        num_train_epochs=epochs,
        warmup_steps=500,
        weight_decay=0.01,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    return model


# %% [markdown] id="XncFTnd_MyZS"
# ## リスト13.12

# %% id="1WFZ4jkLM2wp" colab={"base_uri": "https://localhost:8080/", "height": 382} outputId="9255f815-01cc-4f2c-85ad-cfe6eeff7adf"
import os

king_model_path = os.path.join(os.getcwd(), 'king_model')
prince_model_path = os.path.join(os.getcwd(), 'prince_model')
kingdom_model_path = os.path.join(os.getcwd(), 'kingdom_model')
prince2king_model_path = os.path.join(
    os.getcwd(), 'prince2king_model')

king_dataset = create_king_dataset(df["King"])
king_model = train_king_model(king_model_path, king_dataset)

datasets = create_seq2seq_datasets(df["King"], df["Prince"])
train_dataset_prince, val_dataset_prince = datasets
prince_model = train_seq2seq_model(
    prince_model_path,
    train_dataset_prince,
    val_dataset_prince,
    epochs=6
)

king_and_prince = [f"{k} {p}" for k, p in zip(df["King"], df["Prince"])]
df["King and Prince"] = king_and_prince
train_dataset_kingdom, val_dataset_kingdom = create_seq2seq_datasets(
    df["King and Prince"], df["Kingdom"]
)
kingdom_model = train_seq2seq_model(
    kingdom_model_path,
    train_dataset_kingdom,
    val_dataset_kingdom,
   epochs=6
)

# %% [markdown] id="aGVvACG4M68F"
# ## リスト13.13

# %% id="9o9ryk6wM_u4" colab={"base_uri": "https://localhost:8080/", "height": 906, "referenced_widgets": ["b20832df4ba74230948b5bcdf2426ba4", "7490650b8df94e86b801af13906768bc", "5bd8820b76454149811868122974bfae", "e94021439d2f47269ef96476f71cea93", "fd482688e58c4880aaaf71ae24a83fb8", "02cee6e1e25645c3ac24f6b1400cc6f5", "671957c0838d4df198968bbd9245f003", "f1d8a545df504e379f1c4f6033de30fa", "03a5491c06804b3fa0e5be9aefdb54be", "81a7ce8eb4b145baa2a8328aff10ec31", "5d783e33ec6d45a38349ea9fd1279603"]} outputId="95b8fed5-50c7-42e1-b54d-e79ebc1f89a8"
p2k_data = create_seq2seq_datasets(
    df["Prince"], df["King"])
train_dataset_prince2king, val_dataset_prince2king = p2k_data
prince2king_model = train_seq2seq_model(
    prince2king_model_path,
    train_dataset_prince2king,
    val_dataset_prince2king,
    epochs=6
)

# %% [markdown] id="QEvzljQjNExW"
# ## リスト13.14

# %% id="LJbg-lNkM_63" colab={"base_uri": "https://localhost:8080/", "height": 457, "referenced_widgets": ["c0ea054083294fa78410bfaa7333629f", "bc12ade2d9124673be92767eaccf259c", "868d5daf4546421d9650872367455788", "c138b9896bfd4d27a2058e6eb6a6a8cc", "ef7035503cac438b85fb6b6b46ecab53", "7edb1d0232c64e39bd28e81d32f894e8", "95953358885d440a89d2b54ba4d292b1", "d51c1ed69b6a4fde9287ceefa57ec959", "224e8376558c42e492db67ae639354c5", "893c9e016faa46c783f608065b7d90e9", "6d7b128419ce4239936a41c75b04335a", "42f580a5a6994bf79c3465969f29042f", "a53abc75268e4660afa37f5017868643", "c98a4cd3f1d042e5941934b8f36a6b10", "5b1949556b1648a98490bc038fa20c2c", "3dd832d382254f4eba8780c605d084ea", "0b610fb3f6a7458dbf2b8abf8f536bda", "7f844f85070c49c18175f3b910602ab8", "12375d3258e94baea86e921f8aef953c", "07c30d313ef0476ba84f124082d0adfb", "caa385259c7c493f9c32a06bccda43ef", "51da5272cab44be8a297ee4b9117b22d", "ec1db11cda6a4fb9aa4f1e18a7a73eb2", "4a94b7ac72404c37a665e8bfaeb1d75e", "08a87157a31e49728e84ef6963889df2", "4e1f977594b74bde80e33d34f95b598f", "a458154ca3d847b1b2861ee810fb3ecb", "c7052e750ada435cafe072926daf8da0", "b53bee19d51349b1b8785464ea076de0", "0ad15821779d4a0ca7a8411ebd4c3bd8", "ddcfb7c135d645398a78cf32c90bd677", "aa7579b651954efa9a3352e73bff1417", "e5d667ef172442e7bd8441d1d16871b8", "bce86adf219e494d9a3b3fa6a54e6828", "d1fd91d7a46a40ff988caca7b8fca297", "cfae591244a249839ec4a32c0698c8de", "feff1c9dab154482b7df2cff512a7a05", "0884972cb73c416b9c748cf1236996af", "3d7bc88daf1b4b82a4e3107e6a1e151f", "f99a46e139f9459b8cea1a81e9fa4674", "c0c27f85e3a148b68f9a845b1cbbce57", "afb6c78d5d034e6383909a2b7b81b5bb", "3799ea95132b4381b3d1e604a3d4e03f", "52171f6d3214425fbf6eeefed4cc1c59", "50a8d8defba24e27b23193c64a62d28c", "25eb2ca2a6a14b6b88220c71db5140c8", "3be4e177351047d0956e0518e817a305", "257e8e8497bc4705882684119dde994d", "f9533663b50742f9a5a15daccf73f5d7", "f7c0e87943634e3993c75728f3ae02b8", "ad668d8a8f2a416fa9651ee67d6a6080", "f30a642b59ce4bc89f53b89cc910c390", "7e3eae99f34347cea607c82cc9e4fc44", "bac91c4960054a1bbb51746f7244c2c6", "db5069def10246c9be159f1b80fb67ae", "e69b84ffc48f4dcbbc547f70eacdf756", "c62fcce3e0d7477ab70cae8be955b813", "4ec02c08d8204b10b84bcd3ea6ff8dbb", "405c01e471e0498c809d8de5b0e17cd7", "01fc9ce77e9a4d5daf8f6a8435c82837", "eb79f81a86a14e0ca07fcd021f2f939f", "8f20994296d74ef58ea70afd8d163dbd", "2a1f29d6bf7744ba8299493a0df51953", "0889cde6ea6c479d999d3d27f82112c8", "5cf6410088a5454282f54dca0a38cc95", "4eb01a97c12e4dd38483ba99a2f5f140", "0bf4f6ae3dfb48689ee6ee764cbc200c", "069b8df1f7a24baa945a1aca91317bb9", "fcc22e730c0e42f0824785c6ad7aa9b2", "ef863f282b9946b1a15131b86102b999", "21480c72a77b40b8a89eff7963d0c0c6", "34c3aecf777640c5899d8e4f40a6be6e", "356bc7383f8f45628d89d1266ee61417", "1e60a55d08bc4824b420ec3d9b51acb1", "60594abb7fe8492f9be5b8b856f66cf5", "39536c7e9dc84f5b86b9e56f466a6725", "d7ed5f6ac55a484d87aa3201459de8d9", "1ddbb5268f6e42eeb1b1afa37c68ca35", "1a188caec9474410923ae1a7e26d5d4c", "032dd97392634b838d116295bf4a89ac", "c717737b86fd476a96c0276100bd33c7", "0a69debef1c541b8b00ea99dd97dfb17", "d2390330e2394341bbbba905af4ea602", "99d5f022c0294f579daed4f614d05b8e", "e5e47b11645f4a8d94e2687ca4a7f8ad", "3c886b5705aa4c1299138f3d9110315c", "859f0b62fed6481b8add76ddc0b09d53", "6b69704a26d847778d77d4dbb6245360", "8d312173082341a5af71581411b227e0", "7bcf8a50c96a42e2ab1b99e949d37bbe", "04eed9e018de4bdc8f1582f5c3ec95b1", "b8367fa25ba94348befc0e3716f82294", "8762b1ab0b8b4c2eaa23d372b7b3cb82", "47d11c39a5534ca6869e0a6052aa6aef", "358339502820402da17ae44c36dca799", "05f51fa9a1664733b33f9104e2b6bd16", "e218378b23b743f58d29b5fca91e453f", "5280dff288954b899cd179b353e5f560", "b225425a9e5f451b924fc63381ee7421", "b67e254e1f374c4c984782a77965a137", "e0793fa9495a4bcd98fef7bf90671c4f", "1dad45e0c4b944eba374429c1803bde8", "f5a37187a76b40599e6dc1cce02799a3", "1e94432cb7d14fb684eefec7666103ae", "0f3d1bd094034a9cb16cbc07f84ff63f", "0177b5fa73b843bdab228b1febaa7ec4", "faf37a15c0d24029847e52afdc8018e8", "2b942fcfa7e54ca286fd86b5f68315cf", "4172e46c41ab4a488400b0becd2cff79", "8ce0bf1bb2534908876b78c06084b61e", "bce71a7b1d9445728d142ab1f1433015", "58cf354ec34b4c76b59a5f2b8df3b822", "9bae855d39604fa380116de7b48232e2", "7b0b64d59fa742618558487e49459bcf", "ab8a30f95fba415b85f2938adb157177", "ecc1dde7eaf94305bd024ab54d0e2288", "1f2eb5bb6dd7457e8aebdf4453f50069", "59aebe5fd74e4cf6a1e1a1e46f906d0c", "5029d2bc4c3e4e1bb90515e345889844", "cbfc533b955141b9a5a735382bc041e3", "a37e506607e347908019a0fe9859aae9", "a926245406cd4884ba9c84ca5cb33760", "63fda89fab6e412fbfc664a20b7ed368", "fa3b9ac52f65486e898cbaab4ddb968b", "3f701ff1e2474dd9b495cb31f2fc3c19", "556af8076fd74961a18a977f669f4e5e", "972ef23aecc64f709f67097bf27bb8a0", "bb68ce1e02aa4316a95df63185aec012", "36c10b5d12a24131916289e2f8c599e0", "238d93ab80f847d8ae59c74e5becf5fc", "53a80347f3384896b355dd743a619ab5", "facae74a5fcd4d2da415985e5ff77471"]} outputId="e258906c-1354-4360-a11c-d129133ca6c6"
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, GPT2LMHeadModel,
    PreTrainedModel, BartForConditionalGeneration)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

king_model = AutoModelForCausalLM.from_pretrained(
    "osazuwa/causalLLM-king").to(DEVICE)
prince_model = AutoModelForSeq2SeqLM.from_pretrained(
    "osazuwa/causalLLM-prince").to(DEVICE)
kingdom_model = AutoModelForSeq2SeqLM.from_pretrained(
    "osazuwa/causalLLM-kingdom").to(DEVICE)
prince2king_model = AutoModelForSeq2SeqLM.from_pretrained(
    "osazuwa/causalLLM-prince2king").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer.pad_token = tokenizer.eos_token


# %% [markdown] id="WYS5DwLINUsL"
# ## リスト13.15

# %% id="RkGskmyLNZ9P"
def encode(text:str, device=DEVICE) -> torch.tensor:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    return input_ids

def decode(text_ids: torch.tensor) -> str:
    output = tokenizer.decode(text_ids, skip_special_tokens=True)
    return output

EMPTY_TEXT = torch.tensor(tokenizer.encode("")).unsqueeze(0).to(DEVICE)

def generate_from_model(model: PreTrainedModel,
                        input_sequence: torch.tensor = EMPTY_TEXT,
                        max_length: int = 25,
                        temperature=1.0):
    output = model.generate(
        input_sequence,
        max_length=max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_p=0.9,
    )
    return output

def convert_to_text(output):
    return decode(output[0]).strip().capitalize()


# %% [markdown] id="Xpx2GXTRNkPS"
# ## リスト13.16

# %% id="zOFnSyf8Nw_J"
def compute_log_probs(model, output_sequence):
    if isinstance(model, GPT2LMHeadModel):
        outputs = model(
            input_ids=output_sequence,
            labels=output_sequence
        )
        log_softmax = torch.nn.functional.log_softmax(
            outputs.logits, dim=-1)
        log_probs = log_softmax.gather(2, output_sequence.unsqueeze(-1))
        log_probs = log_probs.squeeze(-1).sum(dim=-1)
    elif isinstance(model, BartForConditionalGeneration):
        outputs = model(
            input_ids=output_sequence,
            labels=output_sequence)
        loss = outputs.loss
        log_probs = -loss * output_sequence.size(1)
    else:
        raise ValueError("Unsupported model type")
    return torch.tensor(log_probs.item())


# %% [markdown] id="0QTUQIReN3-d"
# ## リスト13.17

# %% id="zNF8WKOeN-u_" colab={"base_uri": "https://localhost:8080/"} outputId="aebbe931-df83-4882-a4bd-0c29b5dda088"
king_output = generate_from_model(king_model)
king_statement = convert_to_text(king_output)
print("Generated from king_model:", king_statement)
log_prob_king = compute_log_probs(king_model, king_output)
print("Log prob of generated king text:", log_prob_king)

prince_output = generate_from_model(prince_model, king_output)
prince_statement = convert_to_text(prince_output)
print("Generated from prince_model:", prince_statement)
log_prob_prince = compute_log_probs(prince_model, prince_output)
print("Log prob of generated prince text:", log_prob_prince)

king_prince_statement = king_statement + ". " + prince_statement
king_prince_output = encode(king_prince_statement)
kingdom_output = generate_from_model(kingdom_model, king_prince_output)
kingdom_statement = convert_to_text(kingdom_output)

print("Generated from kingdom model:", kingdom_statement)
log_prob_kingdom = compute_log_probs(kingdom_model, kingdom_output)
print("Log prob of generated kingdom text:", log_prob_kingdom)

king_output_infer = generate_from_model(prince2king_model, prince_output)
king_statement_infer = convert_to_text(king_output_infer)
print("Generated statement from prince2king:", king_statement_infer)
log_prob_prince2king = compute_log_probs(prince2king_model, prince_output)
print("Log prob of generated inference text:", log_prob_prince2king)


# %% [markdown] id="s5c2Ep3qOMrS"
# ## リスト13.18

# %% id="lpMZ1wmTOMMb"
import pyro
from pyro.distributions.torch_distribution \
import TorchDistributionMixin

class TransformerModelDistribution(TorchDistributionMixin):
    def __init__(self, model: PreTrainedModel,
                 input_encoding: torch.tensor = EMPTY_TEXT,
                ):
        super().__init__()
        self.model = model
        self.input_encoding = input_encoding

    def sample(self, sample_shape=torch.Size()):
        output = generate_from_model(
            self.model, self.input_encoding
        )
        return output

    def log_prob(self, value):
        return compute_log_probs(self.model, value)


# %% [markdown] id="ajjWJ5PJb-zo"
# ## リスト13.19

# %% id="FTpATFX9Ax0-" colab={"base_uri": "https://localhost:8080/"} outputId="8f9cebe1-99a9-4d57-d982-56faaa5da76a"
def causalLLM():
    king = pyro.sample(
        "King", TransformerModelDistribution(king_model)
    )
    prince = pyro.sample(
        "Prince", TransformerModelDistribution(
            prince_model, king)
    )
    king_and_prince = torch.cat([king, prince], dim=1)
    kingdom = pyro.sample(
        "Kingdom", TransformerModelDistribution(
            kingdom_model, king_and_prince)
    )
    king_text = convert_to_text(king)
    prince_text = convert_to_text(prince)
    kingdom_text = convert_to_text(kingdom)
    return king_text, prince_text, kingdom_text

for _ in range(2):
    king, prince, kingdom = causalLLM()
    vignette = " ".join([king, prince, kingdom])
    print(vignette)

# %% [markdown] id="c-VivIHDOau8"
# ## リスト13.20

# %% id="P739vZb8OhPV"
import pyro.poutine as poutine
from pyro.distributions import Categorical

PRINCE_STORY = (
    "His courageous Prince takes command, leading "
    "the kingdom's army to victory in battle after battle")
cond_model = pyro.condition(
    causalLLM, {"Prince": encode(PRINCE_STORY)})

def proposal_given_prince():
    prince = encode(PRINCE_STORY)
    king = pyro.sample(
        "King",
        TransformerModelDistribution(prince2king_model, prince)
    )
    king_and_prince = torch.cat([king, prince], dim=1)
    kingdom = pyro.sample(
        "Kingdom",
        TransformerModelDistribution(kingdom_model, king_and_prince)
    )
    vignette = (convert_to_text(king) +
        PRINCE_STORY +
        convert_to_text(kingdom))
    return vignette


# %% [markdown] id="1NOaJbnSQuFn"
# ## リスト13.21

# %% id="RRkUHZuHCnyF"
def process_sample(model, proposal):
    sample_trace = poutine.trace(proposal).get_trace()
    king_text = convert_to_text(sample_trace.nodes['King']['value'])
    kingdom_text = convert_to_text(
        sample_trace.nodes['Kingdom']['value'])
    proposal_log_prob = sample_trace.log_prob_sum()
    replay = poutine.replay(model, trace=sample_trace)
    model_trace = poutine.trace(replay).get_trace()
    model_log_prob = model_trace.log_prob_sum()
    log_importance_weight = model_log_prob - proposal_log_prob
    sample = (king_text, kingdom_text, log_importance_weight)
    return sample


# %% [markdown] id="f6fo10LefJ18"
# ## リスト13.22

# %% id="quSmY9MLQmdB" colab={"base_uri": "https://localhost:8080/"} outputId="5960b53c-5d36-4380-b6d2-d33a2a51e7ce"
def do_importance_resampling(model, proposal, num_samples):
    original_samples = []
    for _ in range(num_samples):
        sample = process_sample(model, proposal)
        original_samples.append(sample)
    unique_samples = list(set(original_samples))
    log_importance_weights = torch.tensor(
        [sample[2] for sample in original_samples])
    resampling_dist = Categorical(logits=log_importance_weights)
    resampled_indices = resampling_dist.sample_n(num_samples)
    samples = pd.DataFrame(
        [unique_samples[i] for i in resampled_indices],
        columns=["King", "Kingdom", "log_importance_weight"]
    )
    samples["Prince"] = PRINCE_STORY
    samples["Distribution"] = "observational"
    return samples[['King', 'Prince', "Kingdom", "Distribution"]]

num_samples = 1000
posterior_samples = do_importance_resampling(
    cond_model, proposal_given_prince, num_samples)


# %% [markdown] id="LuFi4e1TQzGU"
# ## リスト13.23

# %% id="DucANnbgRCcQ" colab={"base_uri": "https://localhost:8080/"} outputId="86ceb19f-4f61-4cd3-9fea-4bf7eb08e042"
intervention_model = pyro.do(
    causalLLM, {"Prince": encode(PRINCE_STORY)})
intervention_samples = pd.DataFrame(
    [intervention_model() for _ in range(num_samples)],
    columns=["King", "Prince", "Kingdom"]
)
intervention_samples["Distribution"] = "interventional"
all_samples = pd.concat(
    [posterior_samples, intervention_samples],
    ignore_index=True
)

# %% [markdown] id="GM5vvthVRaCm"
# ## リスト13.24

# %% id="6DkaDrhyRqFE"
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

kingdom_samples_url = (
    "https://raw.githubusercontent.com/altdeep/causalAI/"
    "master/book/chapter%2013/kingdom_samples.csv")
all_samples = pd.read_csv(kingdom_samples_url)

observational_texts = all_samples[
    all_samples["Distribution"] == "observational"]["Kingdom"]
interventional_texts = all_samples[all_samples[
    "Distribution"] == "interventional"]["Kingdom"]

vectorizer = TfidfVectorizer(stop_words='english')
X_obs = vectorizer.fit_transform(observational_texts)
X_int = vectorizer.transform(interventional_texts)

k = 10
feature_names = vectorizer.get_feature_names_out()
obs_indices = X_obs.sum(axis=0).argsort()[0, -k:][::-1]
int_indices = X_int.sum(axis=0).argsort()[0, -k:][::-1]
combined_indices = np.concatenate((obs_indices, int_indices))
combined_indices = np.unique(combined_indices)

# %% [markdown] id="GvicKlELRu1W"
# ## リスト13.25

# %% id="jp0SgOXsRvBs"
import matplotlib.pyplot as plt

labels = [feature_names[i] for i in combined_indices]
labels, indices = np.unique(labels, return_index=True)
obs_values = np.array(X_obs.sum(axis=0))[0, combined_indices]
int_values = np.array(X_int.sum(axis=0))[0, combined_indices]
obs_values = [obs_values[0][i] for i in indices]
int_values = [int_values[0][i] for i in indices]
combined = list(zip(labels, obs_values, int_values))
sorted_combined = sorted(combined, key=lambda x: (-x[1], x[2]))
labels, obs_values, int_values = zip(*sorted_combined)

width = 0.35
x = np.arange(len(labels))
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, obs_values, width,
                label='Observational', alpha=0.7)
rects2 = ax.bar(x + width/2, int_values, width,
                label='Interventional', alpha=0.7)
ax.set_xlabel('Words')
ax.set_ylabel('TF-IDF Values')
ax.set_title('Top Words in Generated Kingdom Vignettes by TF-IDF Value')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.xticks(rotation=45)
plt.show()
