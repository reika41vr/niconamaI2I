### 各種ライブラリのインポート
import os
import argparse
import random
import requests
import json
import time
import asyncio
import websockets
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import torch
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
from PIL import Image

### ファイル書き込み関数
def write_file(file, text):
    f = open(file, 'w')
    f.write(text)
    f.close()

### 待ち関数
start = False
def wait(t=5):
    global start
    time.sleep(t)
    start = True

# 引数
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--live', type=str,help='ニコ生の放送IDまたはコミュニティID', default="co6073733")
parser.add_argument('-m', '--model', type=str,help='diffusersモデルのパスまたはID', default="SE_V1_A")
parser.add_argument('-n', type=int,help='コメントをまとめて１つのプロンプトにする場合の取得間隔', default=1)
args = parser.parse_args()

### 初期化メッセージの表示
write_file('t2ifile.txt',"Initializing...(wait for a moment, please)")

### コメントを取得したい放送のURLを指定
live_id = args.live
url = "https://live2.nicovideo.jp/watch/"+live_id

### htmlを取ってきてWebSocket接続のための情報を取得
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser") 
embedded_data = json.loads(soup.find('script', id='embedded-data')["data-props"])
url_system = embedded_data["site"]["relive"]["webSocketUrl"]

### websocketでセッションに送るメッセージ
message_system_1 = {"type":"startWatching",
                    "data":{"stream":{"quality":"abr",
                                      "protocol":"hls",
                                      "latency":"low",
                                      "chasePlay":False},
                            "room":{"protocol":"webSocket",
                                    "commentable":True},
                            "reconnect":False}}
message_system_2 ={"type":"getAkashic",
                   "data":{"chasePlay":False}}
message_system_1 = json.dumps(message_system_1)
message_system_2 = json.dumps(message_system_2)

### コメントセッション用のグローバル変数
uri_comment = None
message_comment = None

### Diffusersのinit
model_id = args.model
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.load_textual_inversion("EasyNegative.safetensors", weight_name="EasyNegative.safetensors", token="EasyNegative")

default_prompt = "((masterpiece,best quality)),4k,high resolution,super detailed"
negative_prompt = "EasyNegative,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ((nsfw, nipples, ass, pussy, nude))"
i2i_cmd = "i2i " # i2iをトリガーする接頭語
wildcards = {} # ワイルドカード
for file in os.listdir("wc"):
    f = open("wc/"+file,"r")
    wildcards[os.path.splitext(file)] = f.readlines()

### 視聴セッションとのWebSocket接続関数
async def connect_WebSocket_system():
    global url_system
    global uri_comment
    global message_comment

    ### 視聴セッションとのWebSocket接続を開始
    async with websockets.connect(url_system) as websocket:

        ### 最初のメッセージを送信
        await websocket.send(message_system_1)
        await websocket.send(message_system_2) # これ送らなくても動いてしまう？？
        print("SENT TO THE SYSTEM SERVER: ",message_system_1)
        print("SENT TO THE SYSTEM SERVER: ",message_system_2)

        ### 視聴セッションとのWebSocket接続中ずっと実行
        while True:
            message = await websocket.recv()
            message_dic = json.loads(message)
            print("RESPONSE FROM THE SYSTEM SERVER: ",message_dic)

            ### コメントセッションへ接続するために必要な情報が送られてきたら抽出してグローバル変数へ代入
            if(message_dic["type"]=="room"):
                uri_comment = message_dic["data"]["messageServer"]["uri"]
                threadID = message_dic["data"]["threadId"]
                message_comment = [{"ping": {"content": "rs:0"}},
                                    {"ping": {"content": "ps:0"}},
                                    {"thread": {"thread": threadID,
                                                "version": "20061206",
                                                "user_id": "guest",
                                                "res_from": -150,
                                                "with_global": 1,
                                                "scores": 1,
                                                "nicoru": 0}},
                                    {"ping": {"content": "pf:0"}},
                                    {"ping": {"content": "rf:0"}}]
                message_comment = json.dumps(message_comment)

            ### pingが送られてきたらpongとkeepseatを送り、視聴権を獲得し続ける
            if(message_dic["type"]=="ping"):
                pong = json.dumps({"type":"pong"})
                keepSeat = json.dumps({"type":"keepSeat"})
                await websocket.send(pong)
                await websocket.send(keepSeat)
                print("SENT TO THE SYSTEM SERVER: ",pong)
                print("SENT TO THE SYSTEM SERVER: ",keepSeat)

### コメントセッションとのWebSocket接続関数
async def connect_WebSocket_comment():
    loop = asyncio.get_event_loop()

    global uri_comment
    global message_comment

    ### 視聴セッションがグローバル変数に代入するまで1秒待つ
    await loop.run_in_executor(None, time.sleep, 1)

    ### コメントセッションとのWebSocket接続を開始
    async with websockets.connect(uri_comment) as websocket:

        ### 最初のメッセージを送信
        await websocket.send(message_comment)
        
        ### コメントの受け付けを開始
        write_file('t2ifile.txt',"Waiting for a comment")
        prompt = [default_prompt,]
        
        ### コメントセッションとのWebSocket接続中ずっと実行
        while True:
            message = await websocket.recv()
            message_dic = json.loads(message)
            if start:
                if "chat" in message_dic and message_dic["chat"]["content"].startswith(i2i_cmd) and len(message_dic["chat"]["content"])>len(i2i_cmd):
                    prompt += [message_dic["chat"]["content"][len(i2i_cmd):],]
                if len(prompt)==args.n+1:
                    for i in range(len(prompt[1:])):
                        ### 翻訳
                        prompt[i] = GoogleTranslator(source='ja',target='en').translate(prompt[i])
                        ### プロンプトが空(None)のとき、""を代入
                        if prompt[i] is None:
                            prompt[i] = ""
                        ### ワイルドカードを反映
                        for w in wildcards:
                            if "__"+w+"__" in prompt[i]:
                                s=random.choice(wildcards[w])
                                prompt[i] = prompt[i].replace("__"+w+"__",s.replace("\n",""))
                    ### プロンプトを結合
                    raw_prompt = ','.join(prompt[1:])
                    prompt =",".join(prompt)
                    
                    write_file('t2ifile.txt',"Generating...: "+raw_prompt)

                    ### シードをランダムに決める
                    seed = random.randint(0, 100000)
                    generator = torch.Generator(device).manual_seed(seed)
                    
                    ### I2Iを実行
                    with torch.autocast("cuda"):
                        init_img = Image.open("image1.png")
                        image = pipe(prompt,image=init_img, strength=0.8,negative_prompt=negative_prompt, guidance_scale=8, generator=generator).images[0]
                    write_file('i2imsg.txt',raw_prompt)

                    ### OBSでの表示用に同じ画像を3枚出力
                    image.save("image1.png")
                    image.save("image2.png")
                    image.save("image3.png")
                    prompt = [default_prompt,]

asyncio.new_event_loop().run_in_executor(None,wait)
### asyncioを用いて上で定義した2つのWebSocket実行関数を並列に実行する
loop = asyncio.get_event_loop()
gather = asyncio.gather(
    connect_WebSocket_system(),
    connect_WebSocket_comment())
loop.run_until_complete(gather)