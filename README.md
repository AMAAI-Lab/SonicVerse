
# 🎼 SonicVerse

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/amaai-lab/sonicverse) | [![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-blue?logo=huggingface)](https://huggingface.co/spaces/amaai-lab/SonicVerse) | [![Samples Page](https://img.shields.io/badge/Samples-Page-blue?logo=github)](https://amaai-lab.github.io/SonicVerse/)

An interactive demo for SonicVerse, a music captioning model, allowing users to input audio of up to 10 seconds and generate a natural language caption
that includes a general description of the music as well as music features such as key, instruments, genre, mood / theme, vocals gender.

---

## 🚀 Demo

Check out the live Space here:  
[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-blue?logo=huggingface)](https://huggingface.co/spaces/amaai-lab/SonicVerse)

---

## 🚀 Samples


---

## 📦 Features

✅ Upload a 10 second music clip and get a caption

✅ Upload a long music clip (upto 1 minute for successful demo) to get a long detailed caption for the whole music clip.

---

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/AMAAI-Lab/SonicVerse
cd SonicVerse

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the app
python scripts/app.py
```

---

<!-- ## 📂 File Structure

```
.
├── app.py               # Web app file
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment
├── README.md            # This file
└── src/sonicverse       # Source 
```

--- -->

## 💡 Usage

To use the app:
1. Select audio clip to input 
2. Click the **Generate** button.
3. See the model’s output below.

---

## 🧹 Built With

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio](https://gradio.app/)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [MERT 95M](https://huggingface.co/m-a-p/MERT-v1-95M)
---

<!-- ## ✨ Acknowledgements

- [Model authors or papers you built on]
- [Contributors or collaborators]

---

## 📜 License

This project is licensed under the MIT License / Apache 2.0 / Other.
 -->
