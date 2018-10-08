# Install Docker CE on Ubuntu 16.04 and newer (x86_64/amd64 architectures)

## Setup the Repository

1. Update the **apt** package index:
    ```bash
    $ sudo apt-get update
    ```
2. Install packages:
    ```bash
    $ sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        software-properties-common
    ```
3. Add Docker's official GPG key:
    ```bash
    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    ```
4. Setup the stable repository
    ```bash
    $ sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
    ```

## Install Docker CE
1. Update the **apt** package index
    ```bash
    $ sudo apt-get update
    ```
2. Install the _latest_ version of Docker CE
    ```bash
    $ sudo apt-get install docker-ce
    ```

## More Information of Docker installation

If you are using other OS version, or want to learn more about Docker installation, please view [docker docs](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1).
