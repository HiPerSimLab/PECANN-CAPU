import os
import numpy as np
import imageio


def create_video(cfg, trial: int = 1, fps: int = 24,
                 frames_per_window: int = 21):
    """
    Assemble an MP4 video from the PNG snapshots produced during training.
    """
    methodname = cfg.make_method_name()
    output_path = f'{methodname}_{trial}.mp4'
    writer = imageio.get_writer(output_path, fps=fps)

    domain = np.copy(cfg.domain_st)
    for i in range(cfg.num_win):
        domain[0, 2] = cfg.state_t_array[i]
        domain[1, 2] = cfg.state_t_array[i + 1]
        print(f'[video] assembling window: t in [{domain[0, 2]:.2f}, {domain[1, 2]:.2f}]')

        times = np.linspace(domain[0, 2], domain[1, 2], frames_per_window)
        for t in times:
            png_name = os.path.join(
                cfg.pic_dir,
                f'{methodname}_{trial}_{cfg.state_t_array[i + 1]:.2f}_{t:.2f}s.png'
            )
            image = imageio.imread(png_name)
            writer.append_data(image)

    writer.close()
    print(f'[video] saved to {output_path}')
