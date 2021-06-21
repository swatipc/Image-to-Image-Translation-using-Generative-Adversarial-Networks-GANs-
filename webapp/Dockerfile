FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
ADD https://s3.amazonaws.com/dl-gan-models/model_108000.h5 pix2pix_gan_model.h5
ADD https://dl-gan-models.s3.amazonaws.com/g_model_AtoB_016500.h5 cycle_gan_model.h5
RUN apt-get -y install git
RUN pip install -r requirements.txt
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
EXPOSE 9001
CMD ["python","app.py"]