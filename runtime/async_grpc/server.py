# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import signal
import sys
import asyncio
from concurrent import futures
import argparse
from typing import AsyncGenerator, Callable, Tuple, AsyncIterator, Union

import torch

import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging

import grpc
from grpc import aio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from async_cosyvoice.runtime.async_grpc.utils import convert_audio_tensor_to_bytes, convert_audio_bytes_to_tensor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


import os
import shutil
import subprocess

model_repo_url = "https://www.modelscope.cn/iic/CosyVoice2-0.5B.git"  # 替换为真实的 Git 仓库地址
cache_dir = os.path.expanduser("~/.cache/cosyvoice_0.5b")
link_path = "./pretrained_models/CosyVoice2-0.5B"  # 软链接路径
version_record_file = f"{cache_dir}/._model_version"
config_src = "./async_cosyvoice/CosyVoice2-0.5B"  # 本地配置文件源目录

# 克隆模型（如果未存在）
if not os.path.exists(cache_dir):
    print(f"🔄 模型目录不存在，开始用 git lfs 克隆: {model_repo_url}")
    subprocess.run(["git", "clone", model_repo_url, cache_dir], check=True)
else:
    print("✅ 模型目录已存在，跳过克隆。")

# 记录当前 HEAD 的 commit hash
def get_git_commit_hash(repo_dir):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

current_commit = get_git_commit_hash(cache_dir)

# 判断是否需要更新软链接
need_symlink_update = not os.path.islink(link_path) or os.readlink(link_path) != cache_dir

# 判断模型是否首次下载或有更新（commit hash 变了）
last_model_hash = None
if os.path.exists(version_record_file):
    with open(version_record_file, 'r') as f:
        last_model_hash = f.read().strip()

model_updated = (last_model_hash != current_commit)

# 更新软链接
if need_symlink_update:
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.exists(link_path):
        os.unlink(link_path) if os.path.islink(link_path) else shutil.rmtree(link_path)
    os.symlink(cache_dir, link_path)

# 如果模型更新了，拷贝配置
if model_updated:
    for filename in os.listdir(config_src):
        src_path = os.path.join(config_src, filename)
        dst_path = os.path.join(cache_dir, filename)
        shutil.copy2(src_path, dst_path)
    print(f"✅ 模型更新或首次下载，已拷贝配置文件到: {cache_dir}")
    with open(version_record_file, 'w') as f:
        f.write(current_commit)
else:
    print("✅ 模型未变，跳过配置文件拷贝。")



class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = AsyncCosyVoice2(args.model_dir, load_jit=args.load_jit, load_trt=args.load_trt,
                                             fp16=args.fp16)
        except Exception as e:
            print('no valid model_type! just support AsyncCosyVoice2.')
            raise e
        logging.info('grpc service initialized')

    async def Inference(self, request: cosyvoice_pb2.Request, context: aio.ServicerContext) -> AsyncIterator[
        cosyvoice_pb2.Response]:
        """统一异步流式处理入口"""
        try:
            # 获取处理器和预处理后的参数
            processor, processor_args = await self._prepare_processor(request, request.tts_text)

            # 通过通用处理器生成响应
            async for response in self._handle_generic(request, processor, processor_args):
                yield response

        except Exception as e:
            logging.error(f"Request processing failed: {str(e)}", exc_info=True)
            await context.abort(
                code=grpc.StatusCode.INTERNAL,
                details=f"Processing error: {str(e)}"
            )

    async def StreamInference(self, request_iterator, context: aio.ServicerContext) -> AsyncIterator[
        cosyvoice_pb2.Response]:
        """异步双工流式处理入口，请不要在第一个 request 中包含 tts_text"""
        try:
            async def text_generator(request_iterator):
                async for request in request_iterator:
                    yield request.tts_text

            try:
                # 使用第一个 request 中的参数，构建处理器参数
                first_request = await request_iterator.__anext__()
            except Exception as e:
                return
            # 从后续的请求中 构建 text_gen
            text_gen = text_generator(request_iterator)
            processor, processor_args = await self._prepare_processor(first_request, text_gen)

            # 通过通用处理器生成响应
            async for response in self._handle_generic(first_request, processor, processor_args):
                yield response
        except Exception as e:
            logging.error(f"Request processing failed: {str(e)}", exc_info=True)
            await context.abort(
                code=grpc.StatusCode.INTERNAL,
                details=f"Processing error: {str(e)}"
            )

    async def _prepare_processor(self, request: cosyvoice_pb2.Request, text: Union[str, AsyncGenerator]) -> Tuple[
        Callable, list]:
        """预处理并返回处理器及其参数"""
        match request.WhichOneof('request_type'):
            case 'sft_request':
                return self.cosyvoice.inference_sft, [
                    text,
                    request.sft_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'zero_shot_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.zero_shot_request.prompt_audio
                )
                return self.cosyvoice.inference_zero_shot, [
                    text,
                    request.zero_shot_request.prompt_text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'cross_lingual_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.cross_lingual_request.prompt_audio
                )
                return self.cosyvoice.inference_cross_lingual, [
                    text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'instruct2_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.instruct2_request.prompt_audio
                )
                return self.cosyvoice.inference_instruct2, [
                    text,
                    request.instruct2_request.instruct_text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'instruct2_by_spk_id_request':
                return self.cosyvoice.inference_instruct2_by_spk_id, [
                    text,
                    request.instruct2_by_spk_id_request.instruct_text,
                    request.instruct2_by_spk_id_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'zero_shot_by_spk_id_request':
                return self.cosyvoice.inference_zero_shot_by_spk_id, [
                    text,
                    request.zero_shot_by_spk_id_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case _:
                raise ValueError("Invalid request type")

    async def _handle_generic(
            self,
            request: cosyvoice_pb2.Request,
            processor: Callable,
            processor_args: list
    ) -> AsyncGenerator[cosyvoice_pb2.Response, None]:
        """通用流式处理管道"""
        logging.debug(f"Processing with {processor.__name__}")
        if request.stream:
            # 每一帧当作一个独立的音频返回
            assert request.format in {"", "pcm"}, (
                        "目前流式下，只支持每帧返回一个独立的音频文件(is_frame_independent must be True)，" +
                        "如果需要不同的数据格式，请使用request.format=None返回原始torch.Tensor数据在客户端处理。")
            if request.format in {"", "pcm"}:
                async for model_chunk in processor(*processor_args):
                    audio_bytes = await asyncio.to_thread(
                        convert_audio_tensor_to_bytes,
                        model_chunk['tts_speech'], request.format
                    )
                    yield cosyvoice_pb2.Response(tts_audio=audio_bytes, format=request.format)
            # TODO: 需要在第一帧添加文件头信息，后续的帧直接返回音频数据
            # 在保存音频时，以便使用追加模式写入同一个文件，同时可以使用支持流式播放的音频播放器进行播放。

        else:
            # 服务端合并音频数据后，再编码返回一个完整的音频文件
            audio_data: torch.Tensor = None
            async for model_chunk in processor(*processor_args):
                if audio_data is not None:
                    audio_data = torch.concat([audio_data, model_chunk['tts_speech']], dim=1)
                else:
                    audio_data = model_chunk['tts_speech']

            audio_bytes = await asyncio.to_thread(
                convert_audio_tensor_to_bytes,
                audio_data, request.format
            )
            yield cosyvoice_pb2.Response(tts_audio=audio_bytes, format=request.format)

async def serve(args):
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100M
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]
    server = aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(max_workers=args.max_conc),
        options=options,
        maximum_concurrent_rpcs=args.max_conc
    )
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), server)
    server.add_insecure_port(f'0.0.0.0:{args.port}')
    await server.start()
    logging.info(f"Server listening on 0.0.0.0:{args.port}")
    # 定义一个关闭函数
    async def shutdown(signal, loop):
        logging.info(f"Received exit signal {signal.name}...")
        await server.stop(5)  # 5 秒内优雅关闭
        loop.stop()

    # 捕获信号
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):  # 处理 Ctrl+C 和 kill 信号
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )
    await server.wait_for_termination()

def main(args):
    try:
        asyncio.run(serve(args))
    except asyncio.CancelledError:
        logging.info("Server shutdown complete.")
    except Exception as e:
        logging.error(f"Server encountered an error: {e}")
    finally:
        logging.info("Server has stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max_conc', type=int, default=4)
    parser.add_argument('--model_dir', type=str,
                        default=link_path,
                        help='local path or modelscope repo id')
    parser.add_argument('--load_jit', action='store_true', help='load jit model')
    parser.add_argument('--load_trt', action='store_true', help='load tensorrt model')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    args = parser.parse_args()
    main(args)

    # python server.py --load_jit --load_trt --fp16