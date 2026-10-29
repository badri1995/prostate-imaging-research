from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

LABELS = {
    'radboud': {0: 'background/unknown', 1: 'stroma', 2: 'benign epithelium',
                3: 'Gleason 3 epithelium', 4: 'Gleason 4 epithelium', 5: 'Gleason 5 epithelium'},
    'karolinska': {0: 'background/unknown', 1: 'benign tissue', 2: 'cancerous tissue'},
}

def mask_labels(rgba, provider):
    """PANDA stores integer labels in red; grayscale conversion corrupts them."""
    a = np.asarray(rgba)
    if a.ndim == 3:
        if a.shape[2] == 4 and np.any(a[:, :, 3] != 255):
            raise ValueError('Mask patch includes transparent/out-of-bounds pixels.')
        a = a[:, :, 0]
    if provider not in LABELS:
        raise ValueError(f'Unsupported data provider: {provider}')
    bad = set(np.unique(a).tolist()) - set(LABELS[provider])
    if bad:
        raise ValueError(f'Unexpected mask values {bad}; check provider and mask encoding.')
    return a.astype(np.uint8)

def target_pixels(labels, provider, target):
    if provider not in LABELS:
        raise ValueError(f'Unsupported provider: {provider}')
    cancer = np.isin(labels, [3, 4, 5] if provider == 'radboud' else [2])
    if target == 'cancer':
        return cancer
    if target == 'benign':
        return (labels > 0) & ~cancer
    if target == 'tissue':
        return labels > 0
    raise ValueError('target must be cancer, benign, or tissue')

def physical_patch_size(patch_um, mpp_x, mpp_y):
    if not all(math.isfinite(v) and v > 0 for v in [patch_um, mpp_x, mpp_y]):
        raise ValueError('Patch size and microns per pixel must be positive and finite.')
    return (max(1, round(patch_um / mpp_x)), max(1, round(patch_um / mpp_y)))

def open_pair(dataset_root, image_id, mpp_override=None):
    root = Path(dataset_root)
    image_path = root / 'train_images' / f'{image_id}.tiff'
    mask_path = root / 'train_label_masks' / f'{image_id}_mask.tiff'
    csv_path = root / 'train.csv'
    for p in [image_path, mask_path, csv_path]:
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}. Select a slide with a matching mask or correct DATASET_ROOT.')
    table = pd.read_csv(csv_path, dtype={'image_id': str})
    matches = table[table.image_id == image_id]
    if len(matches) != 1:
        raise ValueError(f'Expected one train.csv row for {image_id}, found {len(matches)}.')
    row = matches.iloc[0].to_dict()
    provider = row['data_provider']
    if provider not in LABELS:
        raise ValueError(f'Unknown provider {provider}.')
    slide = openslide.OpenSlide(str(image_path))
    mask = openslide.OpenSlide(str(mask_path))
    if slide.dimensions != mask.dimensions:
        slide.close(); mask.close()
        raise ValueError('Image and mask dimensions differ. Do not guess an alignment.')
    if mpp_override is None:
        try:
            mpp = tuple(float(slide.properties[k]) for k in ['openslide.mpp-x', 'openslide.mpp-y'])
        except (KeyError, ValueError):
            slide.close(); mask.close()
            raise ValueError('Missing slide pixel calibration. Set MPP_OVERRIDE=(x_um, y_um) from verified metadata.')
    else:
        mpp = tuple(map(float, mpp_override))
    physical_patch_size(500, *mpp)
    info = {'image_id': image_id, 'metadata': row, 'dimensions': list(slide.dimensions),
            'mpp_um': list(mpp), 'calibration_source': 'override' if mpp_override else 'image OpenSlide metadata',
            'mask_alignment': 'PANDA matching ID and identical level-0 pixel grid; image calibration used',
            'mask_mpp_metadata': [mask.properties.get('openslide.mpp-x'), mask.properties.get('openslide.mpp-y')],
            'image_path': str(image_path), 'mask_path': str(mask_path), 'csv_path': str(csv_path)}
    return slide, mask, provider, info

def axis_origins(length, window, stride):
    if window > length:
        raise ValueError('Requested physical patch is larger than the slide.')
    return sorted(set(list(range(0, length-window+1, stride)) + [length-window]))

def choose_patch(slide, mask, provider, mpp, patch_um=500, target='cancer',
                 min_tissue=0.5, min_target=0.1, manual_xy=None):
    w, h = physical_patch_size(patch_um, *mpp)
    W, H = slide.dimensions
    if w*h > 16_000_000:
        raise ValueError('Patch exceeds 16 million pixels; use a smaller PATCH_UM.')
    if w > W or h > H:
        raise ValueError('Patch does not fit inside the slide.')
    level = mask.level_count - 1
    mw, mh = mask.level_dimensions[level]
    overview = mask_labels(mask.read_region((0, 0), level, (mw, mh)), provider)
    # Dimensions rather than rounded downsamples handle pyramid rounding.
    sx, sy = mw/W, mh/H
    rows = []
    for y in axis_origins(H, h, max(1, h//2)):
        for x in axis_origins(W, w, max(1, w//2)):
            a = overview[int(y*sy):max(int(y*sy)+1, math.ceil((y+h)*sy)),
                         int(x*sx):max(int(x*sx)+1, math.ceil((x+w)*sx))]
            tissue, wanted = float(np.mean(a>0)), float(np.mean(target_pixels(a,provider,target)))
            if tissue >= min_tissue and wanted >= min_target:
                rows.append({'x':x, 'y':y, 'tissue_fraction':tissue, 'target_fraction':wanted})
    rows.sort(key=lambda r: (-r['target_fraction'], -r['tissue_fraction'], r['y'], r['x']))
    candidates = pd.DataFrame(rows, columns=['x','y','tissue_fraction','target_fraction'])
    trial = [{'x':int(manual_xy[0]), 'y':int(manual_xy[1])}] if manual_xy is not None else rows
    if not trial:
        raise ValueError('No qualifying region. Try another IMAGE_ID, TARGET, or explicitly adjust coverage limits.')
    for r in trial:
        x, y = r['x'], r['y']
        if x < 0 or y < 0 or x+w > W or y+h > H:
            raise ValueError('Manual level-0 coordinates are out of bounds.')
        labels = mask_labels(mask.read_region((x,y),0,(w,h)), provider)
        coverage = {'tissue_fraction':float(np.mean(labels>0)),
                    'target_fraction':float(np.mean(target_pixels(labels,provider,target)))}
        if coverage['tissue_fraction'] < min_tissue or coverage['target_fraction'] < min_target:
            if manual_xy is not None:
                raise ValueError(f'Manual patch fails coverage limits: {coverage}')
            continue
        rgb = np.array(slide.read_region((x,y),0,(w,h)).convert('RGB'))
        patch_info = {'x_level0':x, 'y_level0':y, 'width_px':w, 'height_px':h,
                      'width_um':w*mpp[0], 'height_um':h*mpp[1], 'target':target,
                      'selection':'manual' if manual_xy is not None else 'highest annotated target fraction on half-patch grid',
                      'label_histogram':{str(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))}, **coverage}
        return rgb, labels, overview, candidates, patch_info
    raise ValueError('Overview candidates failed full-resolution coverage checks. Adjust explicit limits or slide.')

def hematoxylin_signal(rgb):
    signal = np.maximum(rgb2hed(rgb)[:, :, 0], 0)
    lo, hi = np.percentile(signal, [1, 99.5])
    if hi <= lo:
        raise ValueError('Patch lacks stain contrast; choose another region.')
    return np.clip((signal-lo)/(hi-lo), 0, 1)

def mser_candidates(rgb, annotation, provider, mpp, target='cancer', delta=5,
                    min_area_um2=8, max_area_um2=200, min_solidity=0.55):
    """Heuristic nucleus-like regions, NOT validated cell membrane boundaries."""
    signal = hematoxylin_signal(rgb)
    allowed = target_pixels(annotation, provider, target)
    threshold = float(threshold_otsu(signal[allowed])) if np.any(allowed) else 1.0
    dark = signal > threshold
    area_per_pixel = mpp[0]*mpp[1]
    min_px = max(3, math.ceil(min_area_um2/area_per_pixel))
    max_px = max(min_px+1, math.floor(max_area_um2/area_per_pixel))
    detector = cv2.MSER_create()
    detector.setDelta(int(delta))
    detector.setMinArea(min_px)
    detector.setMaxArea(max_px)
    detector.setMaxVariation(0.25)
    detector.setMinDiversity(0.2)
    regions, _ = detector.detectRegions(np.uint8(np.rint((1-signal)*255)))
    accepted = []
    H, W = signal.shape
    for points in regions:
        xs, ys = points[:,0], points[:,1]
        if xs.min()==0 or ys.min()==0 or xs.max()==W-1 or ys.max()==H-1:
            continue  # incomplete objects at the crop boundary
        if allowed[ys,xs].mean() < 0.9 or dark[ys,xs].mean() < 0.6:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(points))
        if hull_area > 0 and len(points)/hull_area < min_solidity:
            continue
        accepted.append((float(signal[ys,xs].mean()), len(points), points))
    accepted.sort(key=lambda v: (-v[1], -v[0], int(v[2][:,1].min()), int(v[2][:,0].min())))
    instances = np.zeros((H,W), dtype=np.int32)
    n = 0
    for _, _, points in accepted:
        xs, ys = points[:,0], points[:,1]
        if np.any(instances[ys,xs]):
            continue  # reject nested and overlapping MSER proposals without trimming geometry
        n += 1
        instances[ys,xs] = n
    params = {'method':'MSER on normalized hematoxylin signal', 'delta':int(delta),
              'min_area_um2':min_area_um2, 'max_area_um2':max_area_um2, 'min_solidity':min_solidity,
              'min_area_px':min_px, 'max_area_px':max_px, 'max_variation':0.25, 'min_diversity':0.2,
              'stain_signal':'nonnegative rgb2hed hematoxylin, percentile 1 to 99.5 normalization',
              'hematoxylin_otsu_threshold':threshold, 'min_annotation_overlap':0.9,
              'min_dark_fraction':0.6, 'raw_regions':len(regions), 'objects':n,
              'overlap_policy':'keep largest qualifying proposal; break ties by mean hematoxylin; reject overlaps',
              'boundary_policy':'exclude objects touching patch edges',
              'biological_status':'unvalidated nucleus-like candidates, not cell membranes'}
    return instances, params

def object_table(instances, annotation, mpp, origin, provider):
    rows = []
    for object_id in np.unique(instances):
        if object_id == 0:
            continue
        ys, xs = np.where(instances == object_id)
        labels, counts = np.unique(annotation[ys,xs],return_counts=True)
        majority = int(labels[np.argmax(counts)])
        rows.append({'object_id':int(object_id), 'area_px':len(xs),
            'area_um2':len(xs)*mpp[0]*mpp[1], 'centroid_x_patch_um':float((xs.mean()+0.5)*mpp[0]),
            'centroid_y_patch_um':float((ys.mean()+0.5)*mpp[1]),
            'centroid_x_level0_px':float(xs.mean()+origin[0]), 'centroid_y_level0_px':float(ys.mean()+origin[1]),
            'majority_panda_label':majority, 'panda_region_name':LABELS[provider][majority],
            'majority_label_fraction':float(counts.max()/counts.sum())})
    return pd.DataFrame(rows)

def svg_paths(instances, mpp):
    records = []
    for object_id in np.unique(instances):
        if object_id == 0:
            continue
        ys,xs = np.where(instances == object_id)
        x0,x1,y0,y1 = int(xs.min()),int(xs.max())+1,int(ys.min()),int(ys.max())+1
        binary = instances[y0:y1,x0:x1] == object_id
        rings = find_contours(np.pad(binary,1).astype(float), 0.5, fully_connected='high')
        paths = []
        for ring in rings:
            points = np.column_stack(((ring[:,1]-0.5+x0)*mpp[0], (ring[:,0]-0.5+y0)*mpp[1]))
            paths.append('M '+' L '.join(f'{x:.6f},{y:.6f}' for x,y in points)+' Z')
        records.append((int(object_id), ' '.join(paths)))
    return records

def export_svgs(instances, mpp, directory, provenance):
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=False)
    H,W = instances.shape
    records = svg_paths(instances,mpp)
    if not records:
        raise ValueError('No objects found: inspect the patch and segmentation parameters.')
    namespace = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', namespace)
    def write(name, paths):
        root = ET.Element(f'{{{namespace}}}svg', {
            'width':f'{W*mpp[0]/1000:.9f}mm', 'height':f'{H*mpp[1]/1000:.9f}mm',
            'viewBox':f'0 0 {W*mpp[0]:.6f} {H*mpp[1]:.6f}'})
        ET.SubElement(root,f'{{{namespace}}}title').text='Unvalidated nucleus-like candidates'
        ET.SubElement(root,f'{{{namespace}}}metadata').text=json.dumps({
            **provenance, 'svg_user_unit':'micrometer', 'axis_direction':'x right, y down',
            'geometry':'pixel-edge isocontours, holes preserved with evenodd fill',
            'three_dimensional_depth':'not assigned'})
        for object_id,d in paths:
            ET.SubElement(root,f'{{{namespace}}}path', {'id':f'object_{object_id:04d}',
                'd':d, 'fill-rule':'evenodd', 'fill':'#277a61', 'stroke':'#173f34', 'stroke-width':'0.2'})
        ET.ElementTree(root).write(dest/name,encoding='utf-8',xml_declaration=True)
    write('combined.svg',records)
    for object_id,path in records:
        write(f'object_{object_id:04d}.svg',[(object_id,path)])
    return len(records)

def overlay(rgb, instances):
    result = rgb.copy()
    for object_id in np.unique(instances):
        if object_id==0:
            continue
        contours,_ = cv2.findContours(np.uint8(instances==object_id),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result,contours,-1,(0,220,130),1)
    return result

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def save_run(directory, rgb, annotation, instances, info, patch_info, params, candidates, comparison):
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(out/'patch.png')
    Image.fromarray(annotation).save(out/'panda_region_labels.png')
    Image.fromarray(overlay(rgb,instances)).save(out/'segmentation_overlay.png')
    np.save(out/'instances.npy',instances)
    table=object_table(instances,annotation,info['mpp_um'],(patch_info['x_level0'],patch_info['y_level0']),info['metadata']['data_provider'])
    if table.empty:
        raise ValueError('No candidate objects; no SVG geometry was exported.')
    table.to_csv(out/'objects.csv',index=False)
    candidates.to_csv(out/'roi_candidates.csv',index=False)
    comparison.to_csv(out/'parameter_comparison.csv',index=False)
    versions={}
    for package in ['numpy','pandas','opencv-python-headless','scikit-image','openslide-python','openslide-bin','matplotlib','cellpose']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    manifest={'source':info, 'patch':patch_info, 'segmentation':params,
              'python':platform.python_version(), 'packages':versions,
              'input_sha256':{k:sha256_file(info[k]) for k in ['image_path','mask_path','csv_path']},
              'status':'technical pipeline tested; biological segmentation requires review',
              'scope':'2D patch and candidate outlines only; no trained model, 3D mesh, or MRI signal generated'}
    count=export_svgs(instances,info['mpp_um'],out/'svg',{'image_id':info['image_id'],'patch':patch_info,'segmentation':params})
    manifest['exported_objects']=count
    manifest['output_sha256']={str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    return manifest,table
