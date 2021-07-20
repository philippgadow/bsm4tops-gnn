# Event generation with MadGraph for BSM4tops

This project containers a Docker setup for MadGraph including a model for top-philic resonances.
It allows for fast generation of truth-level events without detector simulation (or using Delphes as a parametrisation of the ATLAS detector).

The software is provided as a docker image which is hosted on Docker Hub at [philippgadow/mg5-standalone](https://hub.docker.com/repository/docker/philippgadow/mg5-standalone).


## Usage:

### Launching Docker container

On a local machine with Docker installed (e.g. your laptop), execute:
```
mkdir output
docker run -it --rm --volume $PWD:/home/localdir --volume $PWD/output:/var/MG_outputs --volume /tmp:/tmp philippgadow/mg5-standalone

```

### Usage of MadGraph inside container

The Docker container is an isolated environment. The `--volume` command line arguments in the call of `docker run` bind local directories on your machine and allow for accessing them and writing to them from the Docker container.

Therefore, you can use the MadGraph installation (`/home/hep/MG5_aMC_v3_1_1/bin/mg5_aMC`) together with the config file from this project (`commands/simple/generate_bsm4top-13TeV_ttv1_mv1-1500_wv1-auto_t1-1_theta1-0p79.cmnd`) to launch the event generation for the simple formulation of the research problem:

```
# launch mg5 with default event generation for top-philic resonances model
/home/hep/MG5_aMC_v3_1_1/bin/mg5_aMC /home/localdir/commands/simple/generate_bsm4top-13TeV_ttv1_mv1-1500_wv1-auto_t1-1_theta1-0p79.cmnd
```

After leaving the container (`ctrl` + `d`), you can find the output in `output/`:

```
output/proc_bsm4top-evgen-13TeV_ttv1_mv1-1500_wv1-auto_t1-1_theta1-0p79/Events/run_01/unweighted_events.root
```
