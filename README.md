# 🧠 Object Recognition System (OpenCV + KNN, Real-Time Detection)

Современная система распознавания объектов в реальном времени на Python.  
Использует OpenCV, KNN-классификатор и собственный pipeline подготовки данных.

Полный цикл Computer Vision:
подготовка → аугментация → обучение → реальное распознавание

yaml
Копировать код

---

## 🚀 Возможности

✔ Распознавание объектов через веб-камеру
✔ Несколько классов (book, cup, phone, …)
✔ Аугментация изображений (rotate, flip)
✔ Фильтрация фона (background cleanup)
✔ BoundingBox + Label + Confidence%
✔ Автоматические логи (predictions.log)
✔ Автоочистка логов при запуске
✔ Горячие клавиши:
q — выход
s — сохранить кадр
m — сменить режим

yaml
Копировать код

---

## 🗂️ Структура проекта

AI-Recognition-Project/
│
├── data/ # Исходные изображения по классам
│ ├── book/
│ ├── cup/
│ └── phone/
│
├── data_aug/ # Аугментированные изображения (64x64 grayscale)
│
├── main.py # Реальное распознавание через камеру
├── resize.py # Подготовка + аугментация датасета
│
├── model.pkl # Обученная модель KNN
├── predictions.log # Логи работы модели
└── README.md

yaml
Копировать код

---

## ⚙️ Установка

### 🐍 Установите Python и зависимости

```bash
sudo apt install python3 python3-pip
pip install opencv-python scikit-learn numpy joblib
📥 Клонируйте проект
bash
Копировать код
git clone https://github.com/NURJAKS/AI-Recognition-Project.git
cd AI-Recognition-Project
🧪 Подготовка датасета
Разместите изображения в папках:

kotlin
Копировать код
data/
├── book/
├── cup/
├── phone/
└── ...
Все изображения должны быть .jpg / .jpeg.

Запустите аугментацию и подготовку:

bash
Копировать код
python3 resize.py
После выполнения появится:

Копировать код
data_aug/  →  готовый датасет (64×64 grayscale)
▶️ Запуск распознавания
bash
Копировать код
python3 main.py
После запуска система автоматически:

matlab
Копировать код
✓ загрузит model.pkl
✓ активирует веб-камеру
✓ начнёт анализ в реальном времени
✓ покажет рамку, название объекта и confidence %
Если model.pkl отсутствует → модель обучается автоматически.

🧠 Как работает модель (Pipeline)
css
Копировать код
[1] image → grayscale
[2] resize → 64x64
[3] normalize
[4] augmentation → rotate / flip
[5] train KNN classifier
[6] real-time frame capture
[7] bounding box + label + confidence
KNN выбран как:

Копировать код
• быстрый
• интерпретируемый
• отлично работает на маленьких датасетах
📊 Пример работы
pgsql
Копировать код
┌───────────────────────────────────────────────┐
│  object: cup      confidence: 92%             │
│                                               │
│        [█████████  bounding box  ████████]    │
│                                               │
└───────────────────────────────────────────────┘
Рендер происходит в реальном времени через OpenCV.

📌 Возможные улучшения
scss
Копировать код
🔥 Перейти на CNN (TensorFlow / PyTorch)
⚡ Ускорить распознавание через ONNX Runtime
🌐 Добавить Web-интерфейс (Flask + WebRTC)
📦 Docker-контейнеризация
📊 Построение метрик (accuracy, confusion matrix)
💾 Сохранение лучших предсказаний
🔄 Полный ML pipeline: train → evaluate → deploy
🧑‍💻 Автор
yaml
Копировать код
Nurbek Abildaev · 2025  
