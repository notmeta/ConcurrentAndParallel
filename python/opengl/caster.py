import logging
import threading
from typing import List

from opengl import NUMBER_OF_PARTICLES, RAY_CAST_WORKER_THREADS
from particle import Particle


class Caster(threading.Thread):
    def __init__(self, worker_id: int, start_event: threading.Event, particles: List[Particle], rays, canvas):
        super().__init__(name=f"RayCastWorker{worker_id}", target=self._work,
                         args=(start_event, worker_id, particles, rays, canvas))

        self._finished_event = threading.Event()

    def finished(self):
        self._finished_event.wait()
        self._finished_event.clear()

    def _work(self, start_event: threading.Event, worker_id: int, particles: List[Particle], rays, canvas):

        particles_per_thread = NUMBER_OF_PARTICLES / RAY_CAST_WORKER_THREADS
        to_index = int((particles_per_thread * worker_id) - 1) + 1
        from_index = int(to_index - particles_per_thread)

        logging.info(f"caster_id: {worker_id} - casting on indexes {from_index} to {to_index - 1}")

        while True:
            start_event.wait()

            for p in particles[from_index:to_index]:
                for ray in rays:
                    if ray.does_hit_sphere(p.r, p.radius):
                        canvas[ray.X][ray.Y] = p.colour.as_array()
                    else:
                        canvas[ray.X][ray.Y] = [0, 0, 0, 0]

            self._finished_event.set()
