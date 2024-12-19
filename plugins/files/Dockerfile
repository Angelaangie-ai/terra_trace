# Use an official Python runtime as a parent image
FROM python:3.10-bullseye

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=plugin_lib . /tmp/plugin_lib/
RUN pip install /tmp/plugin_lib && rm -rf /tmp/plugin_lib

# Copy the current directory contents into the container at /app
COPY . /app

# Make the container's port 8000 available to the outside world
EXPOSE 8000

# Run app.py and periodic.py when the container launches
CMD ["start_plugin.sh"]
