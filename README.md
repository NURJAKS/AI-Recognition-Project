#  Object Recognition System (OpenCV + KNN, Real-Time Detection)

Современная система распознавания объектов в реальном времени на Python.  
Использует OpenCV, KNN-классификатор и собственный pipeline обработки изображений.

Полный цикл Computer Vision:

```

подготовка → аугментация → обучение → распознавание в реальном времени

```

---

## 🚀 Возможности

- Распознавание объектов через веб-камеру  
- Поддержка нескольких классов (book, cup, phone и др.)  
- Аугментация изображений (rotate, flip)  
- Очистка фона  
- Отрисовка bounding box, имени объекта и confidence %  
- Автологирование (`predictions.log`)  
- Очистка логов при запуске  
- Горячие клавиши:
  - **q** — выход  
  - **s** — сохранить кадр  
  - **m** — сменить режим  

---

## 📁 Структура проекта

```

AI-Recognition-Project/
│
├── data/             # Исходные изображения
│   ├── book/
│   ├── cup/
│   └── phone/
│
├── data_aug/         # Аугментированные 64×64 grayscale изображения
│
├── main.py           # Распознавание объектов (камера)
├── resize.py         # Подготовка и аугментация данных
│
├── model.pkl         # Обученная модель KNN
├── predictions.log   # Логи предсказаний
└── README.md

````

---

## ⚙️ Установка

### 1️⃣ Установка зависимостей

```bash
sudo apt install python3 python3-pip
pip install opencv-python scikit-learn numpy joblib
````

### 2️⃣ Клонирование проекта

```bash
git clone https://github.com/NURJAKS/AI-Recognition-Project.git
cd AI-Recognition-Project
```

---

## 🧪 Подготовка данных

Создайте структуру:

```
data/
├── book/
├── cup/
├── phone/
└── ...
```

Затем выполните аугментацию:

```bash
python3 resize.py
```

После выполнения появится:

```
data_aug/  — готовые 64×64 grayscale изображения
```

---

## ▶️ Запуск распознавания

```bash
python3 main.py
```

Система автоматически:

* загрузит модель
* активирует веб-камеру
* начнёт распознавание
* отобразит:

  * рамку
  * название объекта
  * уровень confidence

Если модель отсутствует — она обучается автоматически.

---

## 🧠 Pipeline модели

```
[1] image → grayscale
[2] resize → 64×64
[3] normalize
[4] augmentation (rotate / flip)
[5] train KNN
[6] capture frame
[7] bounding box + label + confidence
```

KNN выбран потому что:

```
• быстрый
• простой
• интерпретируемый
• отлично работает на небольшом датасете
```

---

## 📊 Пример вывода

```
object: cup      confidence: 92%

┌───────────────────────────────┐
│       [ bounding box ]        │
└───────────────────────────────┘
```

---

## 🔧 Возможные улучшения

* Перейти с KNN → CNN (TensorFlow / PyTorch)
* Ускорение через ONNX Runtime
* Web-интерфейс (Flask + WebRTC)
* Docker-контейнеризация
* Матрица ошибок (confusion matrix), accuracy
* Сохранение лучших предсказаний
* Полный ML pipeline: train → evaluate → deploy

---

## 👨‍💻 Автор

**Nurbek Abildaev — 2025**
