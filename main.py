import pygame
import subprocess
import visualization
import sys
import matplotlib.pyplot as plt

def main():
    pygame.init()
    
    pygame.display.set_caption("Press 'u' to create animation.")
    
    clock = pygame.time.Clock()
    
    print("Zoom in, press q to enhance. Press u to create the animation. Press n to loop the animation.")
    
    plot = visualization.plot_proximity_heatmap("data/fullMap.csv")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        clock.tick(30)

if __name__ == "__main__":
    main()
