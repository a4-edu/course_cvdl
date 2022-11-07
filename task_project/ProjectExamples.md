Список ниже приводит возможные примеры проектов. Сложность проектов и размеры датасетов разная.
Следует воспринимать примеры, как точку старта - до анонса вашего проекта вы должны ознакомиться с темой и удостовериться, что проект выполним в условиях ограниченного времени и вычислительных ресурсов.


1. Fine-Grained Classification

Классификация близких классов грибов на датасете Danish Fungi 20 (или Danish Fungi 20 - Mini) методом metric learning : большой датасет с фото грибов; очень много близких классов, которые не-специалисту сложно отличить.

https://github.com/picekl/DanishFungiDataset, https://arxiv.org/pdf/2103.10107.pdf


2. Video Action Recognition

Классификация активности на видео с помощью Tiny Video Networks на датасете UCF50. Сети для обработки видео обычно довольно тяжелые - но в этой работе авторы из google research с помощью NAS находят сверх-легкие архитектуры, которые работает не сильно хуже больших SOTA моделей.

https://arxiv.org/pdf/1910.06961.pdf, https://www.crcv.ucf.edu/data/UCF50.php


3. Monocular depth estimation

Восстановление карты глубины по RGB изображению с помощью FastDepth на датасете NYU Depth v2

http://fastdepth.mit.edu/


4. Multi-Object Tracking

Решение задачи трекинга с помощью CenterTrack на датасете MOT17

https://arxiv.org/pdf/2004.01177v2.pdf, https://motchallenge.net/data/MOT17/


5. 2D Pose estimation

Определение позы (ключевых точек тела) с помощью CenterPose для амурских тигров (датасет ATRW)

https://github.com/tensorboy/centerpose, https://cvwc2019.github.io/challenge.html


6. Instance segmentation

Сегментация подводного мусорк методом CenterMask на датасете TrashCan.

https://arxiv.org/pdf/1911.06667.pdf, https://conservancy.umn.edu/handle/11299/214865


7. Weakly supervized segmentation

Предсказание масок объектов на основе карты активаций классификатора

https://arxiv.org/pdf/2008.01201.pdf, http://irvlab.cs.umn.edu/resources/suim-dataset)


8. Deep Image Inpainting

Восстановление поврежденных участков фотографий. Есть фотография, есть маска дефектов (того, что нужно закрасить, используя только окружение (контекст).

https://towardsdatascience.com/10-papers-you-must-read-for-deep-image-inpainting-2e41c589ced0, https://arxiv.org/pdf/1905.09010.pdf

9. Camera Pose Estimation

Определение положения камеры в известной локации; можно попробовать Matching In The Dark датасет

https://arxiv.org/abs/2103.09213