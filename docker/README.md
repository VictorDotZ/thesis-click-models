# Окружение для обучения графовых моделей

Контейнер с ROCm 6.2.2 на Ubuntu 24.04. Нужен только графовой части: кликовые
модели из PyClick (CTR, SDBN, VSDBN) считаются на CPU и контейнера не требуют.

## Запуск

```bash
docker compose build rocm
docker compose run --rm rocm-app
```

Пробрасываются `/dev/kfd` и `/dev/dri`, пользователь добавляется в группу
`video` — без этого ROCm не увидит устройство.
