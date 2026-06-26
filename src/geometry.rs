#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: [f32; 3],
    pub d: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageBoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn pixel_to_ray(x: f32, y: f32, intrinsics: CameraIntrinsics) -> [f32; 3] {
    let ray = [
        (x - intrinsics.cx) / intrinsics.fx,
        (y - intrinsics.cy) / intrinsics.fy,
        1.0,
    ];
    normalize(ray)
}

pub fn backproject_depth(x: f32, y: f32, depth_m: f32, intrinsics: CameraIntrinsics) -> [f32; 3] {
    [
        (x - intrinsics.cx) * depth_m / intrinsics.fx,
        (y - intrinsics.cy) * depth_m / intrinsics.fy,
        depth_m,
    ]
}

pub fn estimate_floor_plane(points: &[[f32; 3]]) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }

    let mut centroid = [0.0; 3];
    for point in points {
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let inv = 1.0 / points.len() as f32;
    centroid[0] *= inv;
    centroid[1] *= inv;
    centroid[2] *= inv;

    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    for window in points.windows(3) {
        let a = sub(window[1], window[0]);
        let b = sub(window[2], window[0]);
        let normal = normalize(cross(a, b));
        if !normal.iter().all(|v| v.is_finite()) {
            continue;
        }
        let score = normal[1].abs();
        if score > best_score {
            let normal = if normal[1] < 0.0 {
                [-normal[0], -normal[1], -normal[2]]
            } else {
                normal
            };
            best_score = score;
            best = Some(Plane {
                normal,
                d: -dot(normal, centroid),
            });
        }
    }
    best
}

pub fn depth_at_bbox_contact_region(
    depth_m: &[f32],
    image_width: u32,
    image_height: u32,
    bbox: ImageBoundingBox,
) -> Option<f32> {
    if depth_m.len() != image_width as usize * image_height as usize {
        return None;
    }
    let x0 = bbox.x.min(image_width);
    let x1 = bbox.x.saturating_add(bbox.width).min(image_width);
    let y1 = bbox.y.saturating_add(bbox.height).min(image_height);
    let contact_h = bbox.height.max(1).div_ceil(5).max(1);
    let y0 = y1.saturating_sub(contact_h).max(bbox.y.min(image_height));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }

    let mut values = Vec::new();
    for y in y0..y1 {
        let row = y as usize * image_width as usize;
        for x in x0..x1 {
            let value = depth_m[row + x as usize];
            if value.is_finite() && value > 0.0 {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len <= f32::EPSILON {
        [f32::NAN, f32::NAN, f32::NAN]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_center_points_forward() {
        let intrinsics = CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 40.0,
            width: 100,
            height: 80,
        };
        assert_eq!(pixel_to_ray(50.0, 40.0, intrinsics), [0.0, 0.0, 1.0]);
        assert_eq!(
            backproject_depth(60.0, 40.0, 2.0, intrinsics),
            [0.2, 0.0, 2.0]
        );
    }

    #[test]
    fn contact_depth_uses_lower_bbox_median() {
        let depth = vec![
            1.0, 1.0, 1.0, 1.0, //
            2.0, 2.0, 2.0, 2.0, //
            3.0, 3.0, 9.0, 3.0, //
            4.0, 4.0, 4.0, 4.0,
        ];
        let value = depth_at_bbox_contact_region(
            &depth,
            4,
            4,
            ImageBoundingBox {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
        )
        .unwrap();
        assert_eq!(value, 4.0);
    }
}
