#!/bin/bash

# Define aliases
echo "alias p=python" >> ~/.bashrc
echo "alias cl=clear" >> ~/.bashrc

# Execute the command provided to the Docker container
exec "$@"
