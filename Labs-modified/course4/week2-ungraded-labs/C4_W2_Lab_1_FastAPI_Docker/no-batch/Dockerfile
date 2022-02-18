#The "FROM" instruction allows you to select a pre-existing image as the base for your new image. For example below we load some default images from some source:
	# frolvlad is the username of the author of the image.
	# alpine-miniconda3: This image contains an alpine version of Linux with miniconda with Python 3.
	# python3.7 is just a tag of image, to specify I want the specific python version: 3.7; Tagging is great as it allows you to create different versions of similar images. In this case you could have this same image with a different version of Python such as 3.5.

FROM frolvlad/alpine-miniconda3:python3.7

#Now that you have an environment with Python installed above, it is time to install all of the Python packages/dependencies that your server will depend on, from your local file: requirements.txt at the main folder
COPY requirements.txt .

RUN pip install -r requirements.txt && \
	rm requirements.txt

	# NOTE for above "rm": we remove that requirement.txt file after it is used to install the packages. This is a good practice to always delete uncessary file for image, so that image only include necessary file for server to run. 
	# "&&" allows you to chain two commands in one "RUN" instruction.


# EXPOSE instruction tells Docker to get all its information required during the runtime from a specified Port.
# Since you are coding a web server it is a good idea to leave some documentation about the port that the server is going to listen on. You can do this with the EXPOSE instruction. In this case the server will listen to requests on port 80:

EXPOSE 80
	# Note: This is optional if you are doing the single container web server, but for multi-containers, it is beneficial for inter-container communication, (ref: https://we-are.bookmyshow.com/understanding-expose-in-dockerfile-266938b6a33d)
	# for example: Let’s say we have two containers, a nodejs application and a redis server. Our node app needs to communicate with the redis server for several reasons.
		# For the node app to be able to talk to the redis server, the redis container needs to expose the port.

# Copy server code into docker image
COPY ./app /app
	#Note: first "./app" is source dir from local /app folder; 
		# second "/app" is destination dir in the docker image 

# CMD: This is the command that will be run once a container that uses this image is started
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
	# "unicorn" is an ASGI web server implementation for Python, this is needed to run FastAPI application in a remote server machine (Details are not digged further here)