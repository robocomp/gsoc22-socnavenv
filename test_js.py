import pygame

# Window size
WINDOW_WIDTH    = 300
WINDOW_HEIGHT   = 300


class Controller:
    """ Class to interface with a Joystick """
    def __init__( self, joy_index=0 ):
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick( joy_index )
        self.joystick.init()

    def getAxisValue( self, axis ):
        value = self.joystick.get_axis( axis )
        return value


### initialisation
pygame.init()
window = pygame.display.set_mode( ( WINDOW_WIDTH, WINDOW_HEIGHT ) )
clock  = pygame.time.Clock()
pygame.display.set_caption( "Any Joy?" )    

# Talk to the Joystick
control = Controller()

# Main loop
done = False
while not done:
    joystick_count = pygame.joystick.get_count()
    print(joystick_count)
    pygame.event.pump()
    for event in pygame.event.get():
        if ( event.type == pygame.QUIT ):
            done = True

    # Query the Joystick
    val = control.getAxisValue( 0 )
    # print( "Joystick Axis: " + str( val ) )

    # Update the window, but not more than 60fps
    window.fill( (0,0,0) )
    pygame.display.flip()
    clock.tick_busy_loop(60)

pygame.quit()

