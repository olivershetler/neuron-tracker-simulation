gsutil mb gs://BUCKET_NAME/

gsutil -m cp -r /path/to/your/local/mearec gs://BUCKET-NAME/

https://kompose.io/installation/

https://kompose.io/

kompose convert -f compose.yaml

https://cloud.google.com/sdk/gcloud/reference/auth/configure-docker

gcloud auth configure-docker

docker tag spikeinterface/mountainsort5-base gcr.io/neuron-tracker-simulation/mountainsort5-base:latest

Replace PROJECT_ID and TAG

TAG = :latest

https://docs.docker.com/reference/cli/docker/image/tag/

docker push gcr.io/neuron-tracker-simulation/mountainsort5-base:latest

Regions/Zones:

gcloud container clusters create "simulation-cluster" --num-nodes=3 --zone "us-west1-a" --project [PROJECT_ID]

Regions/Zones:
https://cloud.google.com/compute/docs/regions-zones

.

.
gcloud container clusters get-credentials simulation-cluster --zone us-west1-a --project neuron-tracker-simulation

...
Run ^ AFTER `gcloud container clusters create` finishes

And AFTER ^ finishes (i.e. all above commands). You are then ready to apply the Kubernetes Config:


ENABLE WORKLOAD IDENTITY:

( https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver#requirements )

gcloud container clusters update simulation-cluster  --location=us-west1-a --workload-pool=neuron-tracker-simulation.svc.id.goog


GETTING FUSE CSI DRIVER:
gcloud container clusters update simulation-cluster --update-addons GcsFuseCsiDriver=ENABLED --location=us-west1-a

GET CRIDENTIALS
gcloud container clusters get-credentials simulation-cluster --location=us-west1-a

CREATE SERVICE ACCOUNT (user substitute for automation):
kubectl create serviceaccount neuron-tracker-sim-sa

CREATE IAM POLICY BINDING IN GOOGLE CLOUD (GRANT READ WRITE PERMISSIONS TO THE SERVICE ACCOUNT):
gcloud iam service-accounts add-iam-policy-binding neuron-tracker-sim-sa@neuron-tracker-simulation.iam.gserviceaccount.com --member="serviceAccount:neuron-tracker-simulation.svc.id.goog[default/neuron-tracker-sim-sa]" --role="roles/iam.workloadIdentityUser"

ANNOTATE THE SERVICE ACCOUNT:
kubectl annotate serviceaccount neuron-tracker-sim-sa --namespace default iam.gke.io/gcp-service-account=neuron-tracker-sim-sa@neuron-tracker-simulation.iam.gserviceaccount.com

gcloud container clusters describe simulation-cluster --zone us-west1-a --format="value(workloadIdentityConfig.workloadPool)"

CREATE WORKLOAD IDENTITY POOL:
gcloud iam service-accounts create neuron-tracker-sim-sa --display-name "GCS CSI Service Account"



FIRST:
cd kuberentes
(go into Kuberentes folder you created to store all the converted Kuberentes YAML files)

NEXT:
kubectl apply -f ./
(apply all of the YAML config to create the corresponding resources)

...
kubectl delete pod {POD_NAME}
(can use this to delete the Pod, since the above "apply" is just a proof-of-concept that we can go right into using the converted Kuberentes YAML files); this will update the Pod with the new config

_____

apply -f simulation-a1-l1-deployment.yaml --force

get pods

(copy the name of the pod)

kubectl logs -f POD_NAME -c simulation-a1-l1


gcloud iam service-accounts get-iam-policy neuron-tracker-sim-sa@neuron-tracker-simulation.iam.gserviceaccount.com

