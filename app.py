import json
import threading
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel, get_peft_model

# Qwen3-4B качественнее 0.6B, но требует больше RAM/VRAM.
# Для слабого компьютера установи Qwen/Qwen3-0.6B.
MODEL_NAME = "Qwen/Qwen3-4B"
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "training.jsonl"
OUTPUT_DIR = ROOT / "neurolab-trained"
WEB_DIR = ROOT / "web"
MAX_LENGTH = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

app = FastAPI(title="NeuroLab AI")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

tokenizer = None
model = None
model_lock = threading.RLock()
training_lock = threading.Lock()

status = {
    "running": False,
    "stage": "Модель ещё не загружена",
    "progress": 0,
    "loss": None,
    "error": None,
}


class ChatRequest(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=30)


class ExampleRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    answer: str = Field(min_length=1, max_length=12000)


def read_examples():
    if not DATA_FILE.exists():
        DATA_FILE.write_text("", encoding="utf-8")

    examples = []
    with DATA_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    examples.append({"question": q, "answer": a})
            except (json.JSONDecodeError, AttributeError):
                continue
    return examples


def load_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    adapter_config = OUTPUT_DIR / "adapter_config.json"
    if adapter_config.exists():
        model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    else:
        model = base

    model.to(DEVICE)
    model.eval()


def ensure_model():
    if model is None:
        status["stage"] = "Загрузка базовой модели"
        load_model()
        status["stage"] = "Модель готова к работе"


@app.get("/")
def home():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/status")
def get_status():
    return {
        **status,
        "device": DEVICE,
        "model": MODEL_NAME,
        "examples": len(read_examples()),
        "adapter_saved": (OUTPUT_DIR / "adapter_config.json").exists(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if status["running"]:
        raise HTTPException(409, "Модель сейчас обучается. Попробуй позже.")

    allowed = {"user", "assistant"}
    messages = []
    for item in req.messages[-20:]:
        role = item.get("role")
        content = item.get("content")
        if role in allowed and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:12000]})

    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(400, "Последнее сообщение должно быть вопросом.")

    messages.insert(0, {
        "role": "system",
        "content": (
            "Ты NeuroLab AI — универсальный ИИ-помощник. "
            "Отвечай на русском, если пользователь не просит другой язык. "
            "Помогай с программированием, HTML, Python, играми, математикой, "
            "объяснениями и творческими задачами. Пиши полный рабочий код, "
            "когда его просят. Если не уверен в факте, не выдумывай его. "
            "Учитывай контекст предыдущих сообщений."
        ),
    })

    try:
        with model_lock:
            if status["running"]:
                raise HTTPException(409, "Модель сейчас обучается.")

            ensure_model()
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(DEVICE)

            model.eval()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1800,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.08,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            answer = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        return {"answer": answer or "Не удалось сгенерировать ответ."}

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, str(exc))


@app.post("/api/examples")
def add_example(req: ExampleRequest):
    if status["running"]:
        raise HTTPException(409, "Дождись окончания обучения.")

    item = {
        "question": req.question.strip(),
        "answer": req.answer.strip(),
    }
    with DATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {"ok": True, "examples": len(read_examples())}


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            status["loss"] = round(float(logs["loss"]), 5)

    def on_step_end(self, args, state, control, **kwargs):
        total = max(state.max_steps, 1)
        status["progress"] = min(
            99, int(state.global_step * 100 / total)
        )
        status["stage"] = f"Обучение: {state.global_step}/{total}"


def train_worker():
    global model, tokenizer

    try:
        examples = read_examples()
        if len(examples) < 2:
            raise ValueError("Добавь минимум 2 примера перед обучением.")

        status.update({
            "running": True,
            "stage": "Подготовка обучающих данных",
            "progress": 0,
            "loss": None,
            "error": None,
        })

        with model_lock:
            ensure_model()

            # Используем тот же адаптер, если он уже существует.
            if not isinstance(model, PeftModel):
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ],
                    ),
                )

            model.config.use_cache = False
            model.train()

            records = []
            for item in examples:
                messages = [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                encoded["labels"] = encoded["input_ids"].copy()
                records.append(encoded)

            dataset = Dataset.from_list(records)

            args = TrainingArguments(
                output_dir=str(OUTPUT_DIR / "checkpoints"),
                num_train_epochs=3,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=1e-4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                fp16=DEVICE == "cuda" and DTYPE == torch.float16,
                bf16=DEVICE == "cuda" and DTYPE == torch.bfloat16,
                gradient_checkpointing=DEVICE == "cuda",
                optim="adamw_torch",
                dataloader_pin_memory=DEVICE == "cuda",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                ),
                callbacks=[ProgressCallback()],
            )

            status["stage"] = "Настоящее обучение LoRA"
            trainer.train()

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            model.config.use_cache = True
            model.eval()

            status.update({
                "running": False,
                "stage": "Обучение завершено; адаптер сохранён",
                "progress": 100,
                "error": None,
            })

    except Exception as exc:
        traceback.print_exc()
        status.update({
            "running": False,
            "stage": "Ошибка обучения",
            "error": str(exc),
        })
        if model is not None:
            model.eval()


@app.post("/api/train")
def start_training():
    if not training_lock.acquire(blocking=False):
        raise HTTPException(409, "Обучение уже выполняется.")

    if status["running"]:
        training_lock.release()
        raise HTTPException(409, "Обучение уже выполняется.")

    status.update({
        "running": True,
        "stage": "Запуск обучения",
        "progress": 0,
        "loss": None,
        "error": None,
    })

    def run():
        try:
            train_worker()
        finally:
            training_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "message": "Обучение запущено."}


if __name__ == "__main__":
    import uvicorn
    WEB_DIR.mkdir(exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
