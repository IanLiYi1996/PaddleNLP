# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2020 TsinghuaAI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Many thanks for following projects.
# https://github.com/TsinghuaAI/CPM-Generate
# https://github.com/jm12138/CPM-Generate-Paddle

import sys

import numpy as np
import paddle

from paddlenlp.transformers import GPTChineseTokenizer, GPTForCausalLM, GPTTokenizer
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt-cn": (GPTForCausalLM, GPTChineseTokenizer),
    "gpt": (GPTForCausalLM, GPTTokenizer),
}


class Demo:
    def __init__(self, model_type="gpt-cn", model_name_or_path="gpt-cpm-large-cn", max_new_tokens=10):
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        logger.info("Model loaded.")

    # prediction function
    def predict(self, text):
        ids = self.tokenizer(text)["input_ids"]
        input_ids = paddle.to_tensor(np.array(ids).reshape(1, -1).astype("int64"))
        out = self.model.generate(
            input_ids=input_ids, max_new_tokens=self.max_new_tokens, eos_token_id=self.tokenizer.eol_token_id
        )
        # print(out)
        out = [int(x) for x in out[0].numpy().reshape([-1])]
        logger.info("\n" + text + self.tokenizer.convert_ids_to_string(out))

    # One shot example
    def ask_question_cn(self, question):
        self.predict("问题：中国的首都是哪里？答案：北京。\n问题：%s 答案：" % question)

    def ask_question_en(self, question):
        self.predict("Question: Where is the capital of China? Answer: Beijing. \n Question:%s Answer:" % question)

    # dictation poetry
    def dictation_poetry_cn(self, front):
        self.predict("""默写古诗: 大漠孤烟直，长河落日圆。\n%s""" % front)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gpt-cn":
        demo = Demo("gpt-cn", "gpt-cpm-large-cn")
        demo.ask_question_cn("苹果的CEO是谁?")
        demo.ask_question_cn("中国的成立日期?")
        demo.dictation_poetry_cn("举杯邀明月，")
    else:
        demo = Demo("gpt", "gpt2-medium-en")
        demo.ask_question_en("Who is the CEO of Apple?")
