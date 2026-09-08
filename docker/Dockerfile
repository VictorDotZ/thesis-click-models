FROM ubuntu:24.04

ENV ver="6.2.2"

RUN apt-get update
RUN apt-get install wget gpg -y

RUN mkdir --parents --mode=0755 /etc/apt/keyrings

RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$ver/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/amdgpu.list

RUN apt-get update

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ver noble main" \
    | tee --append /etc/apt/sources.list.d/rocm.list

RUN ["echo", "-e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600"]

RUN apt-get update
RUN apt-get install "rocm${ver}" -y
RUN apt-get autoremove -y

EXPOSE 8188
EXPOSE 22/tcp

CMD ["opt/rocm-6.2.2/bin/rocminfo"]
