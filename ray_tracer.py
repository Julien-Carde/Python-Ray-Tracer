import numpy as np
from PIL import Image
import math
import os
import random
import concurrent.futures
import time

# Image size
WIDTH = 1000
HEIGHT = 1000

# Anti-aliasing
samples_per_pixel = 16  

# Camera settings
camera_pos = np.array([0, 1, -3])
viewport_size = 1  
projection_plane_z = 1

use_depth_of_field = True
focal_distance = 5.0
aperture = 0.05

# Main sphere (blue)
sphere1_center = np.array([0, 0, 3])
sphere1_radius = 1
sphere1_color = np.array([0, 0, 255])  
sphere1_reflectivity = 0.6  
sphere1_shininess = 32    

# Second sphere (red)
sphere2_center = np.array([-3, 0, 7])
sphere2_radius = 1
sphere2_color = np.array([255, 0, 0])  
sphere2_reflectivity = 0.1  
sphere2_shininess = 16    

# Third sphere (green)
sphere3_center = np.array([2.5, 0, 5])
sphere3_radius = 1
sphere3_color = np.array([0, 180, 0])  
sphere3_reflectivity = 0.3  
sphere3_shininess = 128    

# Light properties
light_pos = np.array([-2, 4, -1])
light_intensity = 1.5
light_radius = 0.5   
light_samples = 4    

# Checkerboard floor 
plane_y = -1  
checker_color1 = np.array([255, 255, 255])  
checker_color2 = np.array([50, 50, 50])  
floor_reflectivity = 0.1  
shadow_intensity = 0.3

# Reflection settings
max_reflection_depth = 3  

# Sky settings
use_sky_texture = True
sky_image_file = "sky.jpg"  
fallback_sky_color_top = np.array([135, 206, 235])  
fallback_sky_color_bottom = np.array([240, 248, 255])  

# Multithreading settings
use_multithreading = True
num_threads = None  
def load_sky_image():
    """Load a sky image from file or create a fallback gradient."""
    try:
        # Try to load the sky image from the program folder
        if os.path.exists(sky_image_file):
            print(f"Loading sky image: {sky_image_file}")
            sky_img = Image.open(sky_image_file)

            # Convert to RGB
            if sky_img.mode != "RGB":
                sky_img = sky_img.convert("RGB")

            # Convert to numpy array
            sky_texture = np.array(sky_img)
            print(f"Sky image loaded successfully. Size: {sky_texture.shape}")
            return sky_texture
        else:
            print(f"Sky image file '{sky_image_file}' not found. Using fallback gradient.")
            return None
    except Exception as e:
        print(f"Error loading sky image: {e}. Using fallback gradient.")
        return None

# Try to load the sky texture from file
SKY_TEXTURE = load_sky_image()

def random_point_on_unit_disk():
    while True:
        p = 2.0 * np.array([random.random(), random.random()]) - np.array([1, 1])
        if np.dot(p, p) < 1:
            return p

def sample_sky_color(ray_dir):
    # Determine if ray is going up (into the sky)
    if ray_dir[1] <= 0:
        return np.zeros(3)  # Black if ray is going down

    # Convert ray direction to spherical coordinates for texture mapping
    theta = math.atan2(ray_dir[0], ray_dir[2]) + math.pi  
    phi = math.acos(ray_dir[1])                           

    if SKY_TEXTURE is not None:
        # Map spherical coordinates to texture coordinates
        u = int((theta / (2 * math.pi)) * SKY_TEXTURE.shape[1]) % SKY_TEXTURE.shape[1]
        v = int((phi / math.pi) * SKY_TEXTURE.shape[0]) % SKY_TEXTURE.shape[0]

        # Sample sky texture
        return SKY_TEXTURE[v, u]
    else:
        # Use fallback gradient if sky texture couldn't be loaded
        y_factor = ray_dir[1]
        return fallback_sky_color_bottom * (1 - y_factor) + fallback_sky_color_top * y_factor

def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    #Solve equations
    oc = ray_origin - sphere_center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius**2
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return None  # No intersection

    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    return t1 if t1 > 0 else t2 if t2 > 0 else None  # Return closest intersection

def ray_plane_intersect(ray_origin, ray_dir, plane_y):
    if ray_dir[1] == 0:  # Ray is parallel to the plane
        return None

    t = (plane_y - ray_origin[1]) / ray_dir[1]
    return t if t > 0 else None  # Only return positive intersections

def checkerboard_pattern(point):
    checker_size = 1
    check_x = int(np.floor(point[0] / checker_size)) % 2
    check_z = int(np.floor(point[2] / checker_size)) % 2
    return checker_color1 if (check_x + check_z) % 2 == 0 else checker_color2

def compute_lighting(point, normal, view_dir, light_pos, light_intensity, shininess=32):
    light_dir = light_pos - point
    distance = np.linalg.norm(light_dir)
    light_dir /= distance

    # Ambient component
    ambient_strength = 0.1
    ambient = ambient_strength

    # Diffuse component (Lambertian)
    diffuse = max(np.dot(normal, light_dir), 0)

    # Specular component (Phong)
    reflect_dir = reflect_ray(-light_dir, normal)
    specular_strength = 0.5
    spec = specular_strength * pow(max(np.dot(view_dir, reflect_dir), 0), shininess)

    # Attenuation with distance
    attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance)

    # Combine components
    return (ambient + (diffuse + spec) * attenuation) * light_intensity

def is_in_shadow(point, light_pos):
    shadow_dir = light_pos - point
    shadow_dir /= np.linalg.norm(shadow_dir)  # Normalize the direction

    # Check all spheres for shadows
    # Add a small offset to avoid self-intersection
    shadow_origin = point + 0.01 * shadow_dir

    # Check first sphere
    t_shadow1 = ray_sphere_intersect(shadow_origin, shadow_dir, sphere1_center, sphere1_radius)

    # Check second sphere
    t_shadow2 = ray_sphere_intersect(shadow_origin, shadow_dir, sphere2_center, sphere2_radius)

    # Check third sphere
    t_shadow3 = ray_sphere_intersect(shadow_origin, shadow_dir, sphere3_center, sphere3_radius)

    # Return True if any sphere causes a shadow
    return (t_shadow1 is not None) or (t_shadow2 is not None) or (t_shadow3 is not None)

def is_in_soft_shadow(point, light_pos, light_radius, samples=light_samples):
    shadow_total = 0

    for _ in range(samples):
        # Generate random point on the light
        random_offset = np.random.uniform(-1, 1, 3)
        random_offset = random_offset / np.linalg.norm(random_offset) * light_radius * np.random.random()
        sample_light_pos = light_pos + random_offset

        # Check if this sample is in shadow
        if is_in_shadow(point, sample_light_pos):
            shadow_total += 1

    # Return shadow intensity (0 = no shadow, 1 = full shadow)
    return shadow_total / samples

def reflect_ray(incident, normal):
    return incident - 2 * np.dot(incident, normal) * normal

def trace_ray(ray_origin, ray_dir, depth=0):
    if depth > max_reflection_depth:
        return np.zeros(3)  # Return black if we've exceeded max reflection depth

    # Check for sphere intersections
    t_sphere1 = ray_sphere_intersect(ray_origin, ray_dir, sphere1_center, sphere1_radius)
    t_sphere2 = ray_sphere_intersect(ray_origin, ray_dir, sphere2_center, sphere2_radius)
    t_sphere3 = ray_sphere_intersect(ray_origin, ray_dir, sphere3_center, sphere3_radius)

    # Check for plane intersection
    t_plane = ray_plane_intersect(ray_origin, ray_dir, plane_y)

    # No intersection with anything - return sky color
    if t_sphere1 is None and t_sphere2 is None and t_sphere3 is None and t_plane is None:
        if use_sky_texture:
            return sample_sky_color(ray_dir)
        else:
            return np.zeros(3)  # Return black/background color

    # Find the closest intersection
    closest_t = float('inf')
    intersection_type = None

    if t_sphere1 is not None and t_sphere1 < closest_t:
        closest_t = t_sphere1
        intersection_type = "sphere1"

    if t_sphere2 is not None and t_sphere2 < closest_t:
        closest_t = t_sphere2
        intersection_type = "sphere2"

    if t_sphere3 is not None and t_sphere3 < closest_t:
        closest_t = t_sphere3
        intersection_type = "sphere3"

    if t_plane is not None and t_plane < closest_t:
        closest_t = t_plane
        intersection_type = "plane"

    # Calculate the intersection point
    intersection = ray_origin + closest_t * ray_dir

    # Handle each type of intersection
    if intersection_type == "sphere1":
        # First sphere intersection
        normal = (intersection - sphere1_center) / sphere1_radius

        # Calculate view direction for Phong shading
        view_dir = -ray_dir  # Normalized already

        # Calculate direct lighting
        brightness = compute_lighting(intersection, normal, view_dir, light_pos, light_intensity, sphere1_shininess)
        local_color = np.clip(sphere1_color * brightness, 0, 255)

        # Calculate reflection if needed
        if depth < max_reflection_depth:
            reflection_dir = reflect_ray(ray_dir, normal)
            reflection_origin = intersection + 0.001 * normal
            reflection_color = trace_ray(reflection_origin, reflection_dir, depth + 1)

            # Blend local and reflection colors
            color = (1 - sphere1_reflectivity) * local_color + sphere1_reflectivity * reflection_color
        else:
            color = local_color

        return color

    elif intersection_type == "sphere2":
        # Second sphere intersection
        normal = (intersection - sphere2_center) / sphere2_radius

        # Calculate view direction for Phong shading
        view_dir = -ray_dir  # Normalized already

        # Calculate direct lighting
        brightness = compute_lighting(intersection, normal, view_dir, light_pos, light_intensity, sphere2_shininess)
        local_color = np.clip(sphere2_color * brightness, 0, 255)

        # Calculate reflection if needed
        if depth < max_reflection_depth:
            reflection_dir = reflect_ray(ray_dir, normal)
            reflection_origin = intersection + 0.001 * normal
            reflection_color = trace_ray(reflection_origin, reflection_dir, depth + 1)

            # Blend local and reflection colors
            color = (1 - sphere2_reflectivity) * local_color + sphere2_reflectivity * reflection_color
        else:
            color = local_color

        return color

    elif intersection_type == "sphere3":
        # Third sphere intersection
        normal = (intersection - sphere3_center) / sphere3_radius

        # Calculate view direction for Phong shading
        view_dir = -ray_dir  # Normalized already

        # Calculate direct lighting
        brightness = compute_lighting(intersection, normal, view_dir, light_pos, light_intensity, sphere3_shininess)
        local_color = np.clip(sphere3_color * brightness, 0, 255)

        # Calculate reflection if needed
        if depth < max_reflection_depth:
            reflection_dir = reflect_ray(ray_dir, normal)
            reflection_origin = intersection + 0.001 * normal
            reflection_color = trace_ray(reflection_origin, reflection_dir, depth + 1)

            # Blend local and reflection colors
            color = (1 - sphere3_reflectivity) * local_color + sphere3_reflectivity * reflection_color
        else:
            color = local_color

        return color

    else:  # Floor intersection
        # Floor normal always points up
        normal = np.array([0, 1, 0])

        # Calculate view direction for Phong shading
        view_dir = -ray_dir  # Normalized already

        # Get the checkerboard color at this point
        local_color = checkerboard_pattern(intersection)

        # Check if the point is in soft shadow
        shadow_factor = is_in_soft_shadow(intersection, light_pos, light_radius, light_samples)
        if shadow_factor > 0:
            # Mix the shadow intensity based on the shadow factor
            shadow_amt = shadow_intensity + (1.0 - shadow_intensity) * (1.0 - shadow_factor)
            local_color = (local_color * shadow_amt).astype(np.uint8)

        # Calculate reflection if needed
        if depth < max_reflection_depth and floor_reflectivity > 0:
            reflection_dir = reflect_ray(ray_dir, normal)
            reflection_origin = intersection + 0.001 * normal
            reflection_color = trace_ray(reflection_origin, reflection_dir, depth + 1)

            # Blend local and reflection colors
            color = (1 - floor_reflectivity) * local_color + floor_reflectivity * reflection_color
        else:
            color = local_color

        return color

def render_pixel(x, y):
    color = np.zeros(3)

    for _ in range(samples_per_pixel):
        # Apply anti-aliasing by jittering the ray
        offset_x = (random.random() - 0.5) / WIDTH
        offset_y = (random.random() - 0.5) / HEIGHT

        pixel_x = (x - WIDTH / 2 + offset_x) / WIDTH * viewport_size
        pixel_y = -(y - HEIGHT / 2 + offset_y) / HEIGHT * viewport_size  # Flip to match screen space

        primary_ray_dir = np.array([pixel_x, pixel_y, projection_plane_z])
        primary_ray_dir /= np.linalg.norm(primary_ray_dir)

        if use_depth_of_field:
            # Calculate the focus point
            focus_point = camera_pos + focal_distance * primary_ray_dir

            # Generate random point on lens
            lens_radius = aperture / 2
            p = lens_radius * random_point_on_unit_disk()
            lens_pos = camera_pos + np.array([p[0], p[1], 0])

            # New ray from lens position to focus point
            final_ray_dir = focus_point - lens_pos
            final_ray_dir /= np.linalg.norm(final_ray_dir)

            # Trace the ray with depth of field
            color += trace_ray(lens_pos, final_ray_dir)
        else:
            # Trace the primary ray without depth of field
            color += trace_ray(camera_pos, primary_ray_dir)

    # Average the color samples
    color /= samples_per_pixel

    # Apply gamma correction (simple version)
    color = np.power(color / 255.0, 1/2.2) * 255.0

    # Ensure color values are valid
    return np.clip(color, 0, 255).astype(np.uint8)

def render_row(y):
    row = np.zeros((WIDTH, 3), dtype=np.uint8)
    for x in range(WIDTH):
        row[x] = render_pixel(x, y)
    return row

def render():
    start_time = time.time()
    print("Starting render...")

    if use_multithreading:
        # Use multiprocessing to render in parallel
        print(f"Rendering with multithreading...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            # Map rows to executor
            future_to_row = {executor.submit(render_row, y): y for y in range(HEIGHT)}

            # Create empty image
            img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # Track completed rows for progress reporting
            completed = 0
            total_rows = HEIGHT

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_row):
                y = future_to_row[future]
                try:
                    row = future.result()
                    img[y] = row
                    completed += 1

                    # Report progress
                    if completed % max(1, total_rows // 20) == 0:
                        elapsed = time.time() - start_time
                        progress = completed / total_rows
                        eta = elapsed / progress - elapsed if progress > 0 else 0
                        print(f"Progress: {progress:.1%} [{completed}/{total_rows}] - ETA: {eta:.1f}s")

                except Exception as e:
                    print(f"Error rendering row {y}: {e}")
    else:
        # Render sequentially
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        for y in range(HEIGHT):
            # Print progress every 5%
            if y % max(1, HEIGHT // 20) == 0:
                elapsed = time.time() - start_time
                progress = y / HEIGHT
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"Progress: {progress:.1%} [{y}/{HEIGHT}] - ETA: {eta:.1f}s")

            for x in range(WIDTH):
                img[y, x] = render_pixel(x, y)

    # Rendering complete
    total_time = time.time() - start_time
    print(f"Rendering complete! Total time: {total_time:.1f}s")

    # Save and show image
    image = Image.fromarray(img)
    image.save("raytraced_image.png")
    image.show()

# Run the renderer
if __name__ == "__main__":
    render()