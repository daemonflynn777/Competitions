Трек "Трек коррекция компьютерного зрения".
Решение команды "Котопёс."
Денисов Никита, Ловягин Андрей

Основа решения - нейронная сеть для детекции объектов YOLOv5.
Данная сеть показывает sota-результаты, поэтому была выбрана как надёжное и производительное решение.

В качестве обучающих данный для нейронной сети был вручную создан датасет из 61 картинки: 50 картинок гаек и 11 картинок с рельсовыми зажимами.
Несмотря на крайне малое количество обучающих данных и сильный дисбалланс классов, модель всё равно показывает отличные результаты как на данных за сентябрь и октябрь, так и на данных за ноябрь. Дополнительно, на одной из ноябрьских картинок сеть смогла определить отсуствие болта в важном месте - и это учитывая очень малый размер обучающей выборке.
Разработанное решение оказалось крайне производительным, а при увеличении обучающей выборки можно ожидать поразительных результатов. Данное решение позволит сократить время, уходящее на анализ дефектов, а также издержки - ручной труд будет заменён искусственным интеллектом. Мы не только исключим человеческий фактор и повышаем самое важное - безопасность, то есть возмжный пропуск важного дефекта, но и уменьшим время между появлением дефекта и его ремонтон - нашей решение может работать в реальном временени в виде веб-сервиса 24/7 365 дней в году, а результаты работы нейросети могут быть доступны на любом устройстве: от комьютера до смартфона. Ускорение процесса ремонта позволит снизить издержки на ремонт старых дефектов, который подтверждались вручную лишь спустя долгое время.
Наше решение имеет хороший бизнес-потенциал: оно может быть расширено как на другие железнные дороги России, так и на рынки других стран

! Обучать нейронную сеть не нужно - все веса были уже получены и сохранены в процессе обучения на тренировочной выборке !

Для воспроизведения нашего решения нужно:
1. Открыть файл solution.ipynb
2. Запустить ячейку с кодом под заголовком "Import all necessary libraries"
3. Запустить ячейку с кодом под заголовком "Inference on september and october images"
4. Запустить ячейку с кодом под заголовком "Testing detection on november images"
5. Запустить ячейку с кодом под заголовком "Function for demonstrating the results of detection"
6. Запустить ячейку с кодом под заголовком "September and october images after scoring"
7. Запустить ячейку с кодом под заголовком "November images after scoring"