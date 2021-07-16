FROM nvidia/cuda:11.1.1-base-ubuntu20.04


RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    pip \
 && rm -rf /var/lib/apt/lists/*



RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN echo 1234567

RUN git clone https://github.com/ilia1221/test_core.git
RUN pip install -r test_core/requirements.txt
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11J3ipZ_LWSGH8ZRzIjjAJEU76B8JoESv' -O model_cpu.pt
RUN mv model_cpu.pt test_core/weights/model_cpu.pt



EXPOSE 5000
CMD ["python3", "./test_core/service.py"]

#CMD '/bin/sh'
