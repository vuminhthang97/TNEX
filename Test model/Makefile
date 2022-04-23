URL=140677575285.dkr.ecr.ap-southeast-1.amazonaws.com/tnex-analytics/etl

ecr-login:
	aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 140677575285.dkr.ecr.ap-southeast-1.amazonaws.com

training-image: ecr-login
	docker build -t ${URL}:training --build-arg SCRIPT=training_job.py . && \
	docker push ${URL}:training 

predict-image: ecr-login
	docker build -t ${URL}:predict --build-arg SCRIPT=predict_job.py
	docker push ${URL}:predict

all: training-image predict-image
echo DONE