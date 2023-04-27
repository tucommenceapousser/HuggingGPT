import uuid
import gradio as gr
import re
from diffusers.utils import load_image
import requests
from awesome_chat import chat_huggingface
import os

os.makedirs("public/images", exist_ok=True)
os.makedirs("public/audios", exist_ok=True)
os.makedirs("public/videos", exist_ok=True)

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")

class Client:
    def __init__(self) -> None:
        self.OPENAI_KEY = OPENAI_KEY
        self.HUGGINGFACE_TOKEN = HUGGINGFACE_TOKEN
        self.all_messages = []

    def set_key(self, openai_key):
        self.OPENAI_KEY = openai_key
        return self.OPENAI_KEY

    def set_token(self, huggingface_token):
        self.HUGGINGFACE_TOKEN = huggingface_token
        return self.HUGGINGFACE_TOKEN
    
    def add_message(self, content, role):
        message = {"role":role, "content":content}
        self.all_messages.append(message)

    def extract_medias(self, message):
        # url_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?")
        urls = []
        # for match in url_pattern.finditer(message):
        #     if match.group(0) not in urls:
        #         urls.append(match.group(0))

        image_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png)")
        image_urls = []
        for match in image_pattern.finditer(message):
            if match.group(0) not in image_urls:
                image_urls.append(match.group(0))

        audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav)")
        audio_urls = []
        for match in audio_pattern.finditer(message):
            if match.group(0) not in audio_urls:
                audio_urls.append(match.group(0))

        video_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(mp4)")
        video_urls = []
        for match in video_pattern.finditer(message):
            if match.group(0) not in video_urls:
                video_urls.append(match.group(0))

        return urls, image_urls, audio_urls, video_urls

    def add_text(self, messages, message):
        if not self.OPENAI_KEY or not self.OPENAI_KEY.startswith("sk-") or not self.HUGGINGFACE_TOKEN or not self.HUGGINGFACE_TOKEN.startswith("hf_"):
            return messages, "Please set your OpenAI API key and Hugging Face token first!!!"
        self.add_message(message, "user")
        messages = messages + [(message, None)]
        urls, image_urls, audio_urls, video_urls = self.extract_medias(message)

        for image_url in image_urls:
            if not image_url.startswith("http") and not image_url.startswith("public"):
                image_url = "public/" + image_url
            image = load_image(image_url)
            name = f"public/images/{str(uuid.uuid4())[:4]}.jpg" 
            image.save(name)
            messages = messages + [((f"{name}",), None)]
        for audio_url in audio_urls and not audio_url.startswith("public"):
            if not audio_url.startswith("http"):
                audio_url = "public/" + audio_url
            ext = audio_url.split(".")[-1]
            name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
            response = requests.get(audio_url)
            with open(name, "wb") as f:
                f.write(response.content)
            messages = messages + [((f"{name}",), None)]
        for video_url in video_urls and not video_url.startswith("public"):
            if not video_url.startswith("http"):
                video_url = "public/" + video_url
            ext = video_url.split(".")[-1]
            name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
            response = requests.get(video_url)
            with open(name, "wb") as f:
                f.write(response.content)
            messages = messages + [((f"{name}",), None)]
        return messages, ""

    def bot(self, messages):
        if not self.OPENAI_KEY or not self.OPENAI_KEY.startswith("sk-") or not self.HUGGINGFACE_TOKEN or not self.HUGGINGFACE_TOKEN.startswith("hf_"):
            return messages, {}
        message, results = chat_huggingface(self.all_messages, self.OPENAI_KEY, self.HUGGINGFACE_TOKEN)
        urls, image_urls, audio_urls, video_urls = self.extract_medias(message)
        self.add_message(message, "assistant")
        messages[-1][1] = message
        for image_url in image_urls:
            if not image_url.startswith("http"):
                image_url = image_url.replace("public/", "")
                messages = messages + [((None, (f"public/{image_url}",)))]
            # else:
            #     messages = messages + [((None, (f"{image_url}",)))]
        for audio_url in audio_urls:
            if not audio_url.startswith("http"):
                audio_url = audio_url.replace("public/", "")
                messages = messages + [((None, (f"public/{audio_url}",)))]
            # else:
            #     messages = messages + [((None, (f"{audio_url}",)))]
        for video_url in video_urls:
            if not video_url.startswith("http"):
                video_url = video_url.replace("public/", "")
                messages = messages + [((None, (f"public/{video_url}",)))]
            # else:
            #     messages = messages + [((None, (f"{video_url}",)))]
        # replace int key to string key
        results = {str(k): v for k, v in results.items()}
        return messages, results
    
css = ".json {height: 527px; overflow: scroll;} .json-holder {height: 527px; overflow: scroll;}"
with gr.Blocks(css=css) as demo:
    state = gr.State(value={"client": Client()})
    gr.Markdown("<h1><center>HuggingGPT</center></h1>")
    gr.Markdown("<p align='center'><img src='https://i.ibb.co/qNH3Jym/logo.png' height='25' width='95'></p>")
    gr.Markdown("<p align='center' style='font-size: 20px;'>A system to connect LLMs with ML community. See our <a href='https://github.com/microsoft/JARVIS'>Project</a> and <a href='http://arxiv.org/abs/2303.17580'>Paper</a>.</p>")
    gr.HTML('''<center><a href="https://huggingface.co/spaces/microsoft/HuggingGPT?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space and run securely with your OpenAI API Key and Hugging Face Token</center>''')
    gr.HTML('''<center>Note: Only a few models are deployed in the local inference endpoint due to hardware limitations. In addition, online HuggingFace inference endpoints may sometimes not be available. Thus the capability of HuggingGPT is limited.</center>''')
    if not OPENAI_KEY:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                openai_api_key = gr.Textbox(
                    show_label=False,
                    placeholder="Set your OpenAI API key here and press Enter",
                    lines=1,
                    type="password"
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn1 = gr.Button("Submit").style(full_height=True)

    if not HUGGINGFACE_TOKEN:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                hugging_face_token = gr.Textbox(
                    show_label=False,
                    placeholder="Set your Hugging Face Token here and press Enter",
                    lines=1,
                    type="password"
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn3 = gr.Button("Submit").style(full_height=True)
    

    with gr.Row().style():
        with gr.Column(scale=0.6):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
        with gr.Column(scale=0.4):
            results = gr.JSON(elem_classes="json")


    with gr.Row().style():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter. The url must contain the media type. e.g, https://example.com/example.jpg",
                lines=1,
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn2 = gr.Button("Send").style(full_height=True)
        
    def set_key(state, openai_api_key):
        return state["client"].set_key(openai_api_key)

    def add_text(state, chatbot, txt):
        return state["client"].add_text(chatbot, txt)
    
    def set_token(state, hugging_face_token):
        return state["client"].set_token(hugging_face_token)
    
    def bot(state, chatbot):
        return state["client"].bot(chatbot)

    if not OPENAI_KEY:
        openai_api_key.submit(set_key, [state, openai_api_key], [openai_api_key])
        btn1.click(set_key, [state, openai_api_key], [openai_api_key])

    if not HUGGINGFACE_TOKEN:
        hugging_face_token.submit(set_token, [state, hugging_face_token], [hugging_face_token])
        btn3.click(set_token, [state, hugging_face_token], [hugging_face_token])
    
    txt.submit(add_text, [state, chatbot, txt], [chatbot, txt]).then(bot, [state, chatbot], [chatbot, results])
    btn2.click(add_text, [state, chatbot, txt], [chatbot, txt]).then(bot, [state, chatbot], [chatbot, results])
    

    gr.Examples(
        examples=["Given a collection of image A: /examples/a.jpg, B: /examples/b.jpg, C: /examples/c.jpg, please tell me how many zebras in these picture?",
                "Please generate a canny image based on /examples/f.jpg",
                "show me a joke and an image of cat",
                "what is in the examples/a.jpg",
                "based on the /examples/a.jpg, please generate a video and audio",
                "based on pose of /examples/d.jpg and content of /examples/e.jpg, please show me a new image",
                ],
        inputs=txt
    )

demo.launch()