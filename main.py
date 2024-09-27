import pygame
import subprocess
import runcpp
import heatmap
import sys
import matplotlib.pyplot as plt
import heatmap

def main():
    pygame.init()
    
    pygame.display.set_caption("Press 'u' to create animation.")
    
    clock = pygame.time.Clock()
    
    print("Zoom in, press q to enhance. Press u to create the animation. Press n to loop the animation.")
    
    plot = heatmap.plot_proximity_heatmap("data/fullMap.csv")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        clock.tick(30)

if __name__ == "__main__":
    main()
