#!/bin/bash
python test_with_pogema.py --maps_path maps/random.yaml --num_agents 8+16+24+32+48+64+80 --num_episodes 1 --max_episode_steps 256 --json_output results_random.json
python test_with_pogema.py --maps_path maps/mazes.yaml --num_agents 8+16+24+32+48+64+80 --num_episodes 1 --max_episode_steps 256 --json_output results_mazes.json
python test_with_pogema.py --maps_path maps/warehouse.yaml --num_agents 32+64+96+128+160+192 --num_episodes 128 --max_episode_steps 256 --json_output results_warehouse.json
python test_with_pogema.py --maps_path maps/movingai.yaml --num_agents 64+128+192+256 --num_episodes 1 --max_episode_steps 256 --json_output results_movingai.json