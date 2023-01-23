

def first_pass_leap_frog(x,y,u,v,dt):
    new_x = x - (0.5*u*dt)
    new_y = y - (0.5*v*dt)
    return new_x,new_y

def update_position(x,y,u,v,dt):
    new_x = x + (u*dt)
    new_y = y + (v*dt)
    return new_x,new_y

def update_velocity(charge,mass,u,v,Ex,Ey,dt):
    new_u = u + ((charge*Ex)/mass)
    new_v = v + ((charge*Ey)/mass)
    return new_u,new_v