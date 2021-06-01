FROM lambda-stack:latest

WORKDIR /root/

#COPY .mujoco ./.mujoco 


#RUN pip3 install -U 'mujoco-py<2.1,>=2.0'
#ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-dev
RUN apt-get install -y libosmesa6-dev
RUN apt-get install -y libglew-dev
RUN apt install -y patchelf
RUN pip3 install fasteners Cython cffi glfw imageio gym
#RUN pip3 install -U 'mujoco-py<2.1,>=2.0'
RUN pip3 install install comet_ml
#RUN cd multiagent-particle-envs
RUN pip3 install -e /root/multiagent-particle-envs
#RUN cd ..
# COPY . .

#RUN python3 jmpo_diagnostic.py --dataset_size 100 --eta 100 --ep_len 50 --env 'HalfCheetah-v2' --pi_lr 5e-10 --load_jmpo_model --load_data --iters 300 --eval_freq 10
