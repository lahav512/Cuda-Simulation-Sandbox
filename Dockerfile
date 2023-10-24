# Set base image 
FROM  ubuntu:latest
WORKDIR /app
COPY . .

# Install required packages and Zsh
RUN apt-get update && \
    apt-get install -y zsh git wget curl

# Install Oh-My-Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN echo 'PROMPT="%d $ "' >> ~/.zshrc
RUN zsh -c 'source ~/.zshrc'

# other instructions

# Install dependencies

# Install required packages
# RUN pip install --no-cache-dir -r requirements.txt


# Install CUDA toolkit (nvcc)

# Set aliases
RUN echo "alias cls=clear" >> ~/.zshrc
RUN zsh -c 'source ~/.zshrc'

# Add permissions for executables
RUN chmod +x build
RUN chmod +x run
RUN chmod +x startup

# Set startup script
CMD ["./startup"]
