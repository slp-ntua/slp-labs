# Kaldi setup alternatives

We illustrate two tested alternatives for setting up a working kaldi environment. <\br>
1. [Kaldi Image](https://hub.docker.com/r/kaldiasr/kaldi)
2. Manual (Latest) Instructions

## 1. Docker Image
```bash
docker pull kaldiasr/kaldi
# cpu-based image
docker run -it kaldiasr/kaldi:latest
```

> [!WARNING]
> Mount your code and data as volumes. If you don’t, any files created inside the container will be lost when it stops.

> [!TIP]
> You can use `bind mounts`, `docker volumes`, `docker compose` or other solutions.
