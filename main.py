import pygame
import subprocess
import runcpp
import heatmap
import sys
from closest_position import loop_csv

def generate_data_and_plot(plot):
    if plot:
        x = plot.get_xlim()
        y = plot.get_ylim()
        input1 = f"{x[0]} {x[1]} {y[0]} {y[1]}"
        runcpp.run(input1, 50)
        plot = heatmap.plot_proximity_heatmap("data/zoom.csv")
    return plot

def main():
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press 'u' to create animation.")
    
    clock = pygame.time.Clock()
    
    print("Zoom in, press q to enhance. Press u to create the animation. Press n to loop the animation.")
    
    plot = heatmap.plot_proximity_heatmap("data/fullMap.csv")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Generating new data and creating a new diagram...")
                    plot = generate_data_and_plot(plot)
                    print("Done. Press 'q' again to generate new data and create a new diagram.")
                if event.key == pygame.K_n:
                    print("Doing loop")
                    loop_csv()
            
        # Limit the frame rate
        clock.tick(30)

if __name__ == "__main__":
    main()
