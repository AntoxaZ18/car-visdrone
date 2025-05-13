FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Europe/London
ENV PATH="/root/.poetry/bin:$PATH"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
apt update && apt -y upgrade && \
apt install -y software-properties-common

# Установка Python 3.12 и настройка его как версии по умолчанию
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.12 curl && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
    
# установка poetry
RUN curl -sSL https://install.python-poetry.org | python3 -


# # путь к Poetry добавляем в PATH
ENV PATH="/root/.local/bin:$PATH" 

WORKDIR /app

# копируем зависимости
COPY pyproject.toml poetry.lock ./

#устанавливаем в системный питон
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi && \
    # Очистка кэша Poetry
    poetry cache clear pypi --all && \
    # Удаление временных файлов
    rm -rf /root/.cache

EXPOSE 8888

CMD ["poetry", "run", "ipython"]
