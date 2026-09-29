#!/usr/bin/env python3
"""Faithful Python port of the PanoBasic / Matterport conversion toolbox
(im2Sphere.m + combineViews.m) for stitching Matterport3D skybox cube faces
(and horizontal perspective views) into an equirectangular (ERP) panorama.

Reference implementation: Projection/{im2Sphere,combineViews,warpImageFast}.m and
demo_matterport.m (skybox vx/vy/fov table) of that toolbox.

Skybox face geometry (from demo_matterport.m):
  vx = [-pi/2, -pi/2, 0, pi/2, pi, -pi/2]   # longitude (yaw) of face centre
  vy = [ pi/2,  0,    0, 0,    0, -pi/2]    # latitude  (pitch) of face centre
  fov = pi/2 (+eps)                          # 90 deg per face
  -> skybox0=up, 1=left(-90), 2=front(0), 3=right(+90), 4=back(180), 5=down

Coordinate convention (im2Sphere.m): Y is the forward axis, longitude measured
from +Y, alpha=cos(lat)sin(lon), beta=cos(lat)cos(lon), gamma=sin(lat).

--selftest stitches the toolbox example skybox (data/matterport3d_skybox/, passed as
--skybox-dir) and saves the ERP for visual comparison with the toolbox's result*.png.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

# Matterport skybox face directions & fov (demo_matterport.m, "stitch skybox").
SKYBOX_VX = np.array([-np.pi / 2, -np.pi / 2, 0.0, np.pi / 2, np.pi, -np.pi / 2])
SKYBOX_VY = np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0, -np.pi / 2])
SKYBOX_FOV = np.pi / 2 + 1e-3


def im2sphere(im, hori_fov, sphere_w, sphere_h, vx, vy):
    """Port of im2Sphere.m. Returns (remapped_sphere_img, valid_mask[H,W] bool).

    im: HxW or HxWxC float/uint8 perspective image (camera centre at image
        centre, square-ish pixels). hori_fov: horizontal FOV (rad). vx,vy: view
        direction (lon,lat) of the perspective-view centre (rad).
    """
    if im.ndim == 2:
        im = im[:, :, None]
    imH, imW = im.shape[:2]

    # ERP pixel -> viewing angle (0-indexed; MATLAB used 1-indexed TX,TY).
    tx, ty = np.meshgrid(np.arange(sphere_w), np.arange(sphere_h))
    ang_x = (tx + 0.5 - sphere_w / 2.0) / sphere_w * 2.0 * np.pi
    ang_y = -(ty + 0.5 - sphere_h / 2.0) / sphere_h * np.pi

    R = (imW / 2.0) / np.tan(hori_fov / 2.0)

    # tangent-plane contact point [x0 y0 z0]
    x0 = R * np.cos(vy) * np.sin(vx)
    y0 = R * np.cos(vy) * np.cos(vx)
    z0 = R * np.sin(vy)

    # viewing-ray direction per ERP pixel
    alpha = np.cos(ang_y) * np.sin(ang_x)
    beta = np.cos(ang_y) * np.cos(ang_x)
    gamma = np.sin(ang_y)

    division = x0 * alpha + y0 * beta + z0 * gamma
    # avoid div-by-zero; invalidated below via division<0 and bounds
    with np.errstate(divide="ignore", invalid="ignore"):
        x1 = R * R * alpha / division
        y1 = R * R * beta / division
        z1 = R * R * gamma / division

    vecx = x1 - x0
    vecy = y1 - y0
    vecz = z1 - z0

    # in-plane basis vectors
    vposX = np.array([np.cos(vx), -np.sin(vx), 0.0])  # unit
    c = np.array([x0, y0, z0])
    vposY = np.cross(c, vposX)
    vposY_n = np.linalg.norm(vposY)

    deltaX = vecx * vposX[0] + vecy * vposX[1] + vecz * vposX[2]
    deltaY = (vecx * vposY[0] + vecy * vposY[1] + vecz * vposY[2]) / vposY_n

    # to image pixel coords (0-indexed: MATLAB centre (imW+1)/2 -> (imW-1)/2)
    Px = (deltaX + (imW - 1) / 2.0).astype(np.float32)
    Py = (deltaY + (imH - 1) / 2.0).astype(np.float32)

    valid = (division > 0) & (Px >= 0) & (Px <= imW - 1) & (Py >= 0) & (Py <= imH - 1)

    warped = cv2.remap(
        im.astype(np.float32), Px, Py,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if warped.ndim == 2:
        warped = warped[:, :, None]
    warped[~valid] = 0
    return warped, valid


def combine_views(imgs, vxs, vys, fovs, width, height):
    """Port of combineViews.m: per-face im2sphere + valid-weighted average."""
    nC = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
    acc = np.zeros((height, width, nC), np.float64)
    wei = np.zeros((height, width), np.float64)
    for im, vx, vy, fov in zip(imgs, vxs, vys, fovs):
        sph, valid = im2sphere(im, fov, width, height, vx, vy)
        acc += sph
        wei += valid
    out = acc.copy()
    nz = wei > 0
    out[nz] /= wei[nz][:, None]
    if nC == 1:
        out = out[:, :, 0]
    return out, wei


def stitch_skybox(face_paths, width=2048, height=1024):
    """face_paths: list of 6 skybox image paths in order skybox0..5."""
    imgs = []
    for p in face_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
        if im is None:
            raise FileNotFoundError(p)
        imgs.append(im.astype(np.float32) / 255.0)
    erp, _ = combine_views(imgs, SKYBOX_VX, SKYBOX_VY, [SKYBOX_FOV] * 6, width, height)
    return np.clip(erp * 255.0, 0, 255).astype(np.uint8)  # BGR uint8


def skybox_paths(skybox_dir, uuid):
    return [os.path.join(skybox_dir, f"{uuid}_skybox{i}_sami.jpg") for i in range(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skybox-dir", help="dir containing {uuid}_skybox{0..5}_sami.jpg")
    ap.add_argument("--uuid", help="panorama uuid")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", default="erp.png")
    ap.add_argument("--selftest", action="store_true",
                    help="run on the PanoBasic example skybox (pass its data/matterport3d_skybox "
                         "directory as --skybox-dir) and save for visual check")
    args = ap.parse_args()

    if args.selftest:
        if not args.skybox_dir:
            ap.error("--selftest requires --skybox-dir (the example data/matterport3d_skybox directory)")
        base = args.skybox_dir
        uuid = "6f4d197078d14d6e944ed6533598a6f9"
        paths = skybox_paths(base, uuid)
        erp = stitch_skybox(paths, args.width, args.height)
        out = args.out if args.out != "erp.png" else "scripts/mp3d_benchmark/selftest_skybox_erp.png"
        cv2.imwrite(out, erp)
        print(f"[selftest] wrote {out}  shape={erp.shape}  "
              f"mean={erp.mean():.1f} nonzero={(erp.sum(2)>0).mean()*100:.1f}%")
        return

    paths = skybox_paths(args.skybox_dir, args.uuid)
    erp = stitch_skybox(paths, args.width, args.height)
    cv2.imwrite(args.out, erp)
    print(f"[ok] wrote {args.out} shape={erp.shape}")


if __name__ == "__main__":
    main()
