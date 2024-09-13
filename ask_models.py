from openai import OpenAI
from loguru import logger
from config import MODELS, SEVERS

class ModelConfig:
    def __init__(self):
        # 服务器配置
        self.servers = SEVERS
        
        # 模型配置
        self.models = MODELS

    def get_server_config(self, alias):
        if alias not in self.models:
            raise ValueError(f"模型别名 '{alias}' 未配置。")
        model_name, server_name = self.models[alias]
        server_config = self.servers.get(server_name)
        if not server_config:
            raise ValueError(f"服务器 '{server_name}' 的配置未找到。")
        return model_name, server_config

class OpenAIClientManager:
    def __init__(self, config):
        self.config = config
        self.clients_cache = {}

    def get_client(self, server_config):
        cache_key = (server_config["base_url"], server_config["api_key"])
        if cache_key in self.clients_cache:
            logger.info(f"使用缓存的客户端实例：{cache_key}")
            return self.clients_cache[cache_key]
        else:
            logger.info(f"创建新的客户端实例：{cache_key}")
            client = OpenAI(
                base_url=server_config["base_url"],
                api_key=server_config["api_key"],
            )
            self.clients_cache[cache_key] = client
            return client

    def get_answer(self, question, alias='gemini_flash_s1'):
        '''
        使用OpenAI客户端获取答案

        :param question: 问题
        :param alias: 模型别名
        :return: 答案
        '''
        model_name, server_config = self.config.get_server_config(alias)
        client = self.get_client(server_config)
        logger.info(f"使用的模型：{model_name}")
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"获取答案时出错：{e}")
            raise

def main():
    # model_alias = 'gemini_flash_s1'
    # model_alias = 'gemini_pro_s1'
    # model_alias = 'gemini_1_0_pro_s1'
    # model_alias = 'gemini_flash_s2'
    # model_alias = 'gemini_pro_s2'
    # model_alias = 'gemini_1_0_pro_s2'
    # model_alias = 'mixtral'
    # model_alias = 'gemma2_it'
    # model_alias = 'gemma_it'
    model_alias = 'deepseek'
    # model_alias = 'glm_4'
    # model_alias = 'qwen'
    # model_alias = 'yi_1_5'
    # model_alias = 'ernie_speed_128k'
    # model_alias = 'ernie_speed_8k'
    # model_alias = 'ernie_lite_8k'
    # model_alias = 'ernie_tiny_8k'
    # model_alias = 'yi_spark'
    # model_alias = 'yi_large'
    # model_alias = 'glm_4_flash'
    # model_alias = 'hunyuan_lite'

    config = ModelConfig()
    client_manager = OpenAIClientManager(config)

    question = '你是谁？'
    answer = client_manager.get_answer(question, alias=model_alias)
    print(answer)

if __name__ == '__main__':
    main()