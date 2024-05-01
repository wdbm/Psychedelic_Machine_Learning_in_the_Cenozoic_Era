# Jupyter remote

## general description

The aim here is to express how to use a computer with hardware suited to machine learning (e.g. a GPU) remotely, via a server. For the purposes of this documentation, it is assumed that there is a computer with a GPU (e.g. an XMG P507), a server (e.g. www.example.org) and a local computer which is used to access, via the server, an instance of Jupyter running on the computer with the GPU, and with full access to the GPU. In this documentation, the computer with the GPU is referred to as P507, the server is referred to as the server and the local computer is referred to as X390.

![](https://raw.githubusercontent.com/wdbm/Psychedelic_Machine_Learning_in_the_Cenozoic_Era/master/media/2024-04-24T2045Z.png)

## SSH setup

For all of the computers involved, it may be prudent to ensure that SSH keys have been copied as appropriate, that SSH tunnels are kept alive, and that `tmux` is available:

```Bash
sudo apt install autossh tmux

IFS= read -d '' text << "EOF"
TCPKeepAlive yes
ExitOnForwardFailure yes
ServerAliveInterval 30
Protocol 2,1
EOF
if grep -Fxq "ServerAliveInterval" /etc/ssh/ssh_config; then
    echo "configuration found, not overwriting"
else
    echo "${text}" | sudo tee -a /etc/ssh/ssh_config
fi
```

## P507 (computer with GPU)

Create two shell scripts, one to start and maintain a reverse-SSH tunnel from the server to P507, and one to activate the machine learning infrastructure and thence to launch Jupyter. The variable `$SERVER` is used to store the URL or IP address of the server (e.g. www.example.pro). Both scripts could be run from within `tmux`.

### `reverse_SSH.sh`: a script to start and maintain a reverse-SSH tunnel from the server to the P507

Start a reverse-SSH tunnel from the server to P507, forwarding port 19506 on the server to port 22 on P507.

```Bash
#!/bin/bash
export SERVER='www.example.pro'
export AUTOSSH_POLL=60
export AUTOSSH_GATETIME=30
export AUTOSSH_PORT=0

while true; do
    autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 5" -R 19506:localhost:22 $SERVER
    sleep 1000
done
```

### `run_Jupyter.sh`: a script to activate machine learning infrastructure and launch Jupyter

Activate machine learning infrastructure and launch Jupyter on port 19508.

```Bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate tf
jupyter notebook --no-browser --port=19508
```

## X390 (local computer)

Connect via the server to P507.

```Bash
export SERVER='www.example.org'

ssh -L 19507:localhost:19508 $USER@$SERVER -t ssh -L 19508:localhost:19508 $USER@localhost -p 19506
```

In the first part of the command, X390 port 19507 is connected to server port 19508. The option `-t` allocates a pseudo-TTY which is necessary for running the second SSH command. In the second part of the command, inside the first SSH session, a second SSH session is launched to tunnel from the server to P507, where the server localhost port 19508 is connected to the P507 Jupyter port 19508 via the server reverse-SSH tunnel port 19506.

On X390, open a browser and access the Jupyter instance running on P507:

- <http://localhost:19507>

It may be helpful to get the access token for the running Jupyter instance. This can be done with a command like the following, run (perhaps via SSH) on the P507:

```Bash
source ~/miniconda3/bin/activate
conda activate tf
jupyter notebook list
```
