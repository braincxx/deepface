# FACE SWAP
# WHAT WE USE?
- [DIFFUSERS](https://huggingface.co/docs/diffusers/index)
- [controler_aux](https://github.com/huggingface/controlnet_aux)
- [INSIGHFACE](https://github.com/deepinsight/insightface)
-  [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://github.com/tencent-ailab/IP-Adapter)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)



# Основные Шаги Работы
1. Детекция и Выравнивание Лица
Загрузите изображение с помощью cv2.
Используйте FaceAnalysis из InsightFace для обнаружения лиц на изображении.
Выровняйте лицо по ключевым точкам.
2. Извлечение Эмбеддингов


4. Создание Масок Лица
Создайте маску лица с использованием ключевых точек от InsightFace или MediaPipe FaceMesh.

5. Генерация Результата С Инпейтингом И Адаптером IP-FaceID+
Используйте Stable Diffusion Inpaint Pipeline Legacy вместе с адаптером IP-FaceID+ для генерации результата на основе исходного изображения, маски лица и эмбеддингов лица.



## Installation

```
# install latest diffusers
pip install diffusers==0.22.1

# install ip-adapter
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# download the models
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter

# then you can use the main.py
```

