import os
import re
import pandas as pd
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)

model_loaded = False


def get_available_gpu(threshold=20000):
    # Initialize NVML
    nvmlInit()

    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    available_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024

        if free_memory_mb > threshold:
            available_gpus.append(i)

    # Shutdown NVML
    nvmlShutdown()
    # available_gpus = ['0']

    return available_gpus


def load_model():
    global model_loaded, model, tokenizer
    if not model_loaded:
        available_gpus = get_available_gpu(threshold=15000)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
        print('available_gpus[0]', available_gpus[0])
        import torch
        from transformers import AutoModel, BloomForCausalLM, BloomConfig
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        tokenizer = AutoTokenizer.from_pretrained("/root/workspace/external_data/internlm-7b/lora", trust_remote_code=True)
        '''
        model_config = BloomConfig.from_pretrained(
            '/root/workspace/external_data/7bv5/lora',
        )

        with init_empty_weights():
            model = BloomForCausalLM(
                    model_config,
                ).half()
        model = AutoModel.from_pretrained("model", trust_remote_code=True, device_map='auto').quantize(4).half().cuda()
        print('finished load model!')
        tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
        peft_path = "./lora/adapter_model.bin"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True,
            r=8,
            lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(peft_path), strict=False)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('finished load peft!')
        '''
        from internlm_lora import load_quant
        model = load_quant('/root/workspace/external_data/internlm-7b/lora',
                           '/root/workspace/external_data/internlm-7b/lora/internlm-lora-int4-gptq.bin',
                           4,
                           128,
                           fused_mlp=False,
                           eval=True,
                           act_order=True)

        model_loaded = True
        print('1 model on :', next(model.parameters()).device)
    return model, tokenizer


class GLMPredict(object):
    """自动保存最新模型
    """

    def __init__(self):
        self.model, self.tokenizer = load_model()

    def pred_res(self, Instruction, Input):
        import torch
        prompt = f"Instruction: {Instruction}\n"
        if Input:
            prompt += f"Input: {Input}\n"
        prompt += "Answer: "
        with torch.no_grad():
            ids = self.tokenizer.encode(prompt)
            input_ids = torch.LongTensor([ids]).to('cuda')
            out = self.model.generate(input_ids=input_ids, max_length=1500, do_sample=False, temperature=0)
            out_text = self.tokenizer.decode(out[0])
            # answer = out_text.replace(prompt, "").replace("\nEND", "").strip()
            answer = re.findall('Answer:\s*([^\n]+)', out_text)
            answer = '' if len(answer) < 1 else answer[0]
        return answer

    def get_result(self, project_name, field_en, field, context):
        prompt = ''
        '''
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=project_name)
        row = medical_logic.loc[medical_logic['字段英文名称'] == field_en].to_dict('records')[0]
        if row['值域类型'] == '多选':
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择提到的所有内容。"""
        elif row['值域类型'] == '单选':
            prompt = f"##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择1个。"
        elif row['值域类型'] == '提取':
            row['字段名'] = row['字段名'].replace('大小1','大小')
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值"""
        else:
            return ''
        '''
        prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值"""
        print('prompt', prompt)
        res = self.pred_res(prompt, context)
        print('res', res)
        return res


if __name__ == '__main__':
    fd_pred = GLMPredict()
    data_dict = {
        "project":
        "肾上腺-瑞金",
        "field_en":
        "CUARD",
        "field":
        "主动脉根部内径",
        "raw_text":
        "检查途径：经体表  图像等级：乙  检查项目：二维     M型     彩色    多普勒（脉冲式    连续式）一、M型主要测值（单位mm)：                            名称          测量值    正常参考值    主动脉根部内径      38        20-37    左房内径            44        19-40    左室舒张末期内径    52        35-56    左室收缩末期内径    33        20-37    室间隔厚度          13        6-11    左室后壁厚度        13        6-11   二、二维超声心动图描述：1.左房增大,主动脉根部内径增宽。2.各心瓣膜未见明显增厚，开放不受限。3.左室壁增厚，静息状态下左室各节段收缩活动未见明显异常。三、彩色多普勒超声描述:1.各心瓣膜示轻度微主动脉瓣反流。2.舒张期经二尖瓣口血流频谱：E/A＜1。舒张期经二尖瓣口血流：E=63cm/s,A=91cm/s。3.房、室间隔水平未见明显异常分流信号。四、左心功能测定:         名称                    测量值    左室舒张末期容量（ml）        129     左室收缩末期容量（ml）        45     每搏输出量（ml）              84     左室短轴缩短率（%）           36     左室射血分数（%）             65 五、组织多普勒检查：    二尖瓣瓣环水平：室间隔侧： e'=5cm/s， E/e'=12.6。                    左室侧壁： e'=13cm/s,  E/e'=4.9。"
    }
    res = fd_pred.get_result(data_dict["project"], data_dict["field_en"], data_dict["field"], data_dict["raw_text"])
