"""A deliberately limited vocabulary of understandable, image-affecting settings.

Unknown tags are never prettified automatically: a field needs a known meaning
and a useful display value before it belongs in the comparison UI.
"""
from fractions import Fraction
import json
import math

# Friendly name, preferred source tags, plain-language help, optional unit.
FIELDS = (
    ('Exposure time', ('ExposureTime',), 'How long the shutter lets light reach the sensor.', 's'),
    ('Aperture', ('FNumber', 'Aperture'), 'Lens opening. A lower f-number lets in more light and reduces the depth in focus.', 'f'),
    ('ISO sensitivity', ('ISO', 'PhotographicSensitivity', 'ISOSpeedRatings', 'ISOSetting'), 'Sensor sensitivity. Higher settings can increase noise.', 'number'),
    ('Exposure compensation', ('ExposureCompensation', 'ExposureBiasValue'), 'Brightness adjustment relative to the camera’s automatic exposure.', 'EV'),
    ('Exposure mode', ('ExposureProgram', 'ExposureMode'), 'Whether shutter speed and aperture were set manually or automatically.', ''),
    ('Light metering', ('MeteringMode',), 'Which parts of the scene the camera uses to choose exposure.', ''),
    ('Focal length', ('FocalLength',), 'Lens focal length, which determines how wide or zoomed-in the view is.', 'mm'),
    ('Lens', ('LensModel', 'LensID'), 'The lens used to take the photograph.', ''),
    ('Focus mode', ('FocusMode', 'AFMode'), 'Whether focus is set manually or by autofocus.', ''),
    ('Autofocus area', ('AFAreaModeSetting', 'AFAreaMode'), 'The area of the frame used for autofocus.', ''),
    ('Focus tracking', ('AFTracking',), 'Whether autofocus follows a moving subject.', ''),
    ('Focus distance', ('SubjectDistance', 'FocusDistance'), 'Distance from the camera to the subject in focus.', 'm'),
    ('Estimated focus distance', ('FocusDistance2',), 'Approximate lens focus distance inferred from camera metadata. For Sony, ExifTool uses lens position and focal length. Useful for comparing focus between photos; not an accurate measurement of camera-to-root distance.', 'm'),
    ('White balance', ('WhiteBalance',), 'Color adjustment for the lighting, so neutral objects appear neutral.', ''),
    ('White balance red gain', ('RedBalance',), 'Multiplier applied to red relative to green (green = 1). These gains can change between photos even when white balance stays on Auto.', 'gain'),
    ('White balance blue gain', ('BlueBalance',), 'Multiplier applied to blue relative to green (green = 1). A change alters the color balance even when the camera reports the same white-balance mode.', 'gain'),
    ('White balance warmth', ('WBWarmth',), 'Fine adjustment along the blue-to-amber axis. Neutral means no shift.', 'warmth'),
    ('White balance tint', ('WBTint',), 'Fine adjustment along the green-to-magenta axis. Neutral means no shift.', 'tint'),
    ('White balance fine adjustment', ('WBFineAdjustment',), 'Additional camera adjustment to the selected white-balance preset. Zero means unchanged; the scale depends on the camera.', 'number'),
    ('Color tint filter', ('ColorTintFilter',), 'The camera’s extra color filter along the green-to-magenta axis.', 'tint'),
    ('Contrast adjustment', ('ContrastAdjustment',), 'Camera contrast adjustment: negative softens the difference between dark and light; positive strengthens it.', 'number'),
    ('Saturation adjustment', ('SaturationAdjustment',), 'Camera color-strength adjustment: negative reduces color; positive increases it.', 'number'),
    ('Sharpness adjustment', ('SharpnessAdjustment',), 'Camera edge-sharpening adjustment relative to the selected picture style.', 'number'),
    ('Hue adjustment', ('HueAdjustment',), 'Camera adjustment that shifts the colors in the photograph.', 'number'),
    ('Color temperature', ('ColorTemperature',), 'White-balance temperature. Lower values suit warmer lighting.', 'K'),
    ('Flash', ('Flash', 'FlashAction'), 'Whether the flash fired for this photograph.', ''),
    ('Flash mode', ('FlashMode',), 'How the camera is set to use flash.', ''),
    ('Flash compensation', ('FlashExposureComp', 'FlashExposureCompensation'), 'Adjustment to the amount of flash light.', 'EV'),
    ('Contrast', ('Contrast',), 'Strength of the difference between light and dark tones.', ''),
    ('Saturation', ('Saturation',), 'Strength of the colors.', ''),
    ('Sharpness', ('Sharpness',), 'How much the camera emphasizes edges.', ''),
    ('Brightness adjustment', ('Brightness',), 'Brightness adjustment applied by the camera.', 'number'),
    ('Color style', ('CreativeStyle', 'PictureStyle', 'FilmMode', 'ColorMode'), 'The camera’s chosen color and tone treatment.', ''),
    ('Picture profile', ('PictureProfile',), 'The camera’s profile for rendering color and brightness.', ''),
    ('Picture effect', ('PictureEffect', 'PictureEffect2'), 'A special visual effect applied by the camera.', ''),
    ('Scene mode', ('SceneMode', 'SceneCaptureType'), 'Camera settings tailored to a type of scene.', ''),
    ('Color space', ('ColorSpace',), 'The range of colors used to encode the photograph.', ''),
    ('Image quality', ('Quality',), 'The camera’s quality setting, which can affect compression and detail.', ''),
    ('High dynamic range', ('HDRSetting', 'HDR'), 'Combines exposures to retain more detail in bright and dark areas.', ''),
    ('Shadow and highlight adjustment', ('DynamicRangeOptimizer', 'ActiveDLighting'), 'Adjusts dark and bright regions to retain visible detail.', ''),
    ('Long-exposure noise reduction', ('LongExposureNoiseReduction',), 'Reduces noise in photographs taken with long exposures.', ''),
    ('High-ISO noise reduction', ('HighISONoiseReduction', 'NoiseReduction'), 'Reduces noise at high sensitivity, potentially smoothing fine detail.', ''),
    ('Multi-frame noise reduction', ('MultiFrameNoiseReduction',), 'Combines several frames to reduce noise.', ''),
    ('Image stabilization', ('ImageStabilization', 'Anti-Blur', 'VibrationReduction'), 'Compensates for camera shake.', ''),
    ('Dark-corner correction', ('VignettingCorrection', 'PeripheralIlluminationCorr'), 'Brightens corners darkened by the lens.', ''),
    ('Color-fringe correction', ('LateralChromaticAberration', 'ChromaticAberrationCorrection'), 'Reduces colored fringes along high-contrast edges.', ''),
    ('Lens distortion correction', ('DistortionCorrectionSetting', 'DistortionCorrection'), 'Corrects curved lines caused by lens distortion.', ''),
    ('Digital zoom', ('DigitalZoomRatio',), 'Magnification created by cropping or resampling the image.', '×'),
    ('Sensor crop', ('APS-CSizeCapture',), 'Whether only the central area of the sensor is used.', ''),
    ('Electronic first shutter curtain', ('ElectronicFrontCurtainShutter',), 'Starts the exposure electronically to reduce shutter vibration.', ''),
    ('Skin smoothing', ('SoftSkinEffect',), 'Smooths skin texture, which can also remove fine detail.', ''),
    ('Orientation', ('Orientation', 'CameraOrientation'), 'How the photograph should be rotated for display.', ''),
)

FIELD_HELP = {name: help_text for name, _, help_text, _ in FIELDS}

# Standard EXIF codes used by the Pillow fallback. ExifTool already decodes them.
_EXIF_CODES = {
    'WhiteBalance': {0: 'Auto', 1: 'Manual'},
    'ExposureProgram': {1: 'Manual', 2: 'Auto', 3: 'Aperture priority', 4: 'Shutter priority',
                        5: 'Creative', 6: 'Action', 7: 'Portrait', 8: 'Landscape'},
    'ExposureMode': {0: 'Auto', 1: 'Manual', 2: 'Exposure bracketing'},
    'MeteringMode': {1: 'Average', 2: 'Center-weighted', 3: 'Spot', 4: 'Multi-spot', 5: 'Multi-segment', 6: 'Partial'},
    'Contrast': {0: 'Normal', 1: 'Low', 2: 'High'},
    'Saturation': {0: 'Normal', 1: 'Low', 2: 'High'},
    'Sharpness': {0: 'Normal', 1: 'Soft', 2: 'Hard'},
    'SceneCaptureType': {0: 'Standard', 1: 'Landscape', 2: 'Portrait', 3: 'Night'},
    'ColorSpace': {1: 'sRGB', 2: 'Adobe RGB'},
    'Orientation': {1: 'Normal', 2: 'Mirrored horizontally', 3: 'Rotated 180°',
                    4: 'Mirrored vertically', 5: 'Mirrored and rotated 270°',
                    6: 'Rotated 90° clockwise', 7: 'Mirrored and rotated 90°',
                    8: 'Rotated 90° counterclockwise'},
}


def _display_value(key, value, unit):
    value = str(value).strip()
    if (not value or value.lower().startswith(('unknown', 'undefined', 'n/a', 'unrecognized', 'unclassified'))
            or 'SHA-256' in value or value.startswith(('{', '[', 'base64:', '(Binary'))):
        return None
    group, tag = key.rsplit(':', 1)
    if unit == 'm' and value.lower() in {'inf', 'infinity'}:
        return 'Infinity'
    if unit:
        number = value.removeprefix('f/').removesuffix(unit).strip()
        try:
            numeric = float(Fraction(number))
        except (ValueError, ZeroDivisionError):
            return value if unit == 'K' and value.lower() == 'auto' else None
        if not math.isfinite(numeric):
            return None
        if unit in {'warmth', 'tint'}:
            if numeric == 0:
                return 'Neutral'
            positive, negative = ('amber', 'blue') if unit == 'warmth' else ('magenta', 'green')
            return f'{abs(numeric):g} toward {positive if numeric > 0 else negative}'
        if unit == 'gain':
            if numeric <= 0:
                return None
            return f'{numeric:.6f}'.rstrip('0').rstrip('.') + ' ×'
        if unit == 's':
            number = str(Fraction(number).limit_denominator(1000000)) if 0 < numeric < 1 else f'{numeric:g}'
        else:
            number = f'{numeric:g}'
        return f'f/{number}' if unit == 'f' else number if unit == 'number' else f'{number} {unit}'
    try:
        code = int(value)
    except ValueError:
        return value
    if group in {'EXIF', 'ExifIFD', 'IFD0'}:
        if tag == 'Flash':
            return 'Fired' if code & 1 else 'Did not fire'
        return _EXIF_CODES.get(tag, {}).get(code)
    # Manufacturer-specific numeric codes are not meaningful without a decoder.
    return None


def photo_settings(raw):
    """Select one understandable value per setting, preferring standard EXIF."""
    candidates = {}
    for key, value in _with_adjustments(raw).items():
        if ':' not in key:
            continue
        group, tag = key.rsplit(':', 1)
        if group in {'IFD1', 'System', 'File', 'InteropIFD', 'GPS', 'ThumbnailIFD'}:
            continue
        priority = (0 if group in {'ExifIFD', 'EXIF', 'IFD0'} else
                    3 if group == 'Derived' else 2 if group == 'Composite' else 1)
        candidates.setdefault(tag, []).append((priority, key, value))
    settings = {}
    for name, tags, _, unit in FIELDS:
        for tag in tags:
            for _, key, value in sorted(candidates.get(tag, [])):
                display = _display_value(key, value, unit)
                if display is not None:
                    settings[name] = display
                    break
            if name in settings:
                break
    return settings


def _numbers(value, count):
    """Read a documented numeric tuple, accepting ExifTool text or JSON arrays."""
    try:
        if isinstance(value, str):
            value = json.loads(value) if value.startswith('[') else value.split()
        numbers = [float(item) for item in value]
        if len(numbers) == count and all(math.isfinite(item) for item in numbers):
            return numbers
    except (TypeError, ValueError):
        pass
    return None


def _with_adjustments(raw):
    """Translate documented tuples, never proprietary undecoded camera codes."""
    result = dict(raw)
    # ExifTool normally supplies Composite red/blue gains. Also handle readers
    # that expose only RGB levels, using green as the common reference.
    for key, value in sorted(raw.items()):
        if key.rsplit(':', 1)[-1] == 'WB_RGBLevels':
            levels = _numbers(value, 3)
            if levels and all(level > 0 for level in levels):
                red, green, blue = levels
                result['Derived:RedBalance'] = str(red / green)
                result['Derived:BlueBalance'] = str(blue / green)
                break
    # Sony documents positive AB as amber and positive GM as magenta.
    for tag in ('WBShiftAB_GM_Precise', 'WBShiftAB_GM'):
        shift = _numbers(raw.get('Sony:' + tag), 2)
        if shift:
            result['Derived:WBWarmth'], result['Derived:WBTint'] = map(str, shift)
            break
    for source, target in (
        ('ColorCompensationFilter', 'ColorTintFilter'),
        ('WhiteBalanceFineTune', 'WBFineAdjustment'),
        ('Contrast', 'ContrastAdjustment'),
        ('Saturation', 'SaturationAdjustment'),
        ('Sharpness', 'SharpnessAdjustment'),
    ):
        if 'Sony:' + source in raw:
            result['Derived:' + target] = raw['Sony:' + source]
    return result
