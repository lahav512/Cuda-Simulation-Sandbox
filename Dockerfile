# Set base image 
FROM cuda-projects:latest as base

# FROM ubuntu:latest
WORKDIR /app
COPY . .
# SHELL ["/bin/sh", "-c"]

# Update repositories
# RUN apt-get update
FROM base

# Install required packages and Zsh
# RUN apt-get update -y

# Install Oh-My-Zsh
#     apt-get install -y zsh curl git
# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
# SHELL ["/bin/zsh", "-c"]

# other instructions

# Install dependencies

# Add permissions for executables
RUN chmod +x build run startup

# Set startup script
CMD ["./startup"]
