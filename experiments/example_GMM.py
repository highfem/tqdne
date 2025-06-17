from multiprocessing import Pool, cpu_count

import numpy as np
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.mfd import ArbitraryMFD
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.source.characteristic import CharacteristicFaultSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.valid import gsim
from tqdm import tqdm


def calculate_gmfs(mag, rupture_aratio, strike, dip, rake, hypo, imts, Vs30, gmps):
    gmpes = []
    for gmm in gmps:
        gmpes.append(gsim(gmm))
    hypocenter = Point(hypo[0], hypo[1], hypo[2])

    planar_surface = PlanarSurface.from_hypocenter(
        hypoc=hypocenter,
        msr=WC1994(),
        mag=mag,
        aratio=rupture_aratio,
        strike=strike,
        dip=dip,
        rake=rake,
    )

    imtls = {s: [0] for s in imts}

    src = CharacteristicFaultSource(
        source_id=1,
        name="rup",
        tectonic_region_type="Active Shallow Crust",
        mfd=ArbitraryMFD([mag], [0.01]),
        temporal_occurrence_model=PoissonTOM(50.0),
        surface=planar_surface,
        rake=rake,
    )

    ruptures = [r for r in src.iter_ruptures()]

    jb_distances = np.linspace(1, 200, 300)

    rupture = ruptures[0]

    bottom_edge = Line([rupture.surface.bottom_left, rupture.surface.bottom_right])
    bottom_edge = bottom_edge.resample_to_num_points(3)
    mid_point = bottom_edge[1]
    mid_point.depth = 0.0

    locs = [
        mid_point.point_at(
            horizontal_distance=d, vertical_increment=0, azimuth=rupture.surface.strike + 90.0
        )
        for d in jb_distances
    ]

    site_collection = SiteCollection(
        [Site(location=loc, vs30=Vs30, vs30measured=True, z1pt0=40.0, z2pt5=1.0) for loc in locs]
    )

    context_maker = ContextMaker("*", gmpes, {"imtls": imtls})
    ctxs = context_maker.from_srcs(src, site_collection)
    gms = context_maker.get_mean_stds(
        ctxs
    )  # 4 values (0=median, 1=std_total, 2=std_intra, 3=std_inter), then G=2 gsims, M=2 IMTs, 1 scenario = magnitude

    return gms, jb_distances


def calculate_gmfs_distance(mag, rupture_aratio, strike, dip, rake, hypo, imts, Vs30, gmps):
    gmpes = []
    for gmm in gmps:
        gmpes.append(gsim(gmm))
    hypocenter = Point(hypo[0], hypo[1], hypo[2])

    planar_surface = PlanarSurface.from_hypocenter(
        hypoc=hypocenter,
        msr=WC1994(),
        mag=mag,
        aratio=rupture_aratio,
        strike=strike,
        dip=dip,
        rake=rake,
    )

    imtls = {s: [0] for s in imts}

    src = CharacteristicFaultSource(
        source_id=1,
        name="rup",
        tectonic_region_type="Active Shallow Crust",
        mfd=ArbitraryMFD([mag], [0.01]),
        temporal_occurrence_model=PoissonTOM(50.0),
        surface=planar_surface,
        rake=rake,
    )

    ruptures = [r for r in src.iter_ruptures()]

    rupture = ruptures[0]
    jb_distances = np.linspace(1, 250, 250)

    bottom_edge = Line([rupture.surface.bottom_left, rupture.surface.bottom_right])
    bottom_edge = bottom_edge.resample_to_num_points(3)
    mid_point = bottom_edge[1]
    mid_point.depth = 0.0

    locs = [
        mid_point.point_at(
            horizontal_distance=d, vertical_increment=0, azimuth=rupture.surface.strike + 90.0
        )
        for d in jb_distances
    ]

    site_collection = SiteCollection(
        [Site(location=loc, vs30=Vs30, vs30measured=True, z1pt0=40.0, z2pt5=1.0) for loc in locs]
    )

    context_maker = ContextMaker("*", gmpes, {"imtls": imtls})
    ctxs = context_maker.from_srcs(src, site_collection)
    gms = context_maker.get_mean_stds(
        ctxs
    )  # 4 values (0=median, 1=std_total, 2=std_intra, 3=std_inter), then G=2 gsims, M=2 IMTs, 1 scenario = magnitude

    return gms, jb_distances


def gmrotdpp_withPG(
    acceleration_x,
    time_step_x,
    acceleration_y,
    time_step_y,
    periods,
    percentile,
    damping=0.05,
    units="cm/s/s",
    method="Nigam-Jennings",
):
    """
    modified from gmrotdpp to also return gmrotdpp(PGA, PGV and PGD)
    This is much faster than gmrotdpp_slow
    """
    from smtk.intensity_measures import (
        equalise_series,
        get_response_spectrum,
        rotate_horizontal,
    )

    if (percentile > 100.0 + 1e-9) or (percentile < 0.0):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(
        acceleration_x, time_step_x, periods, damping, units, method
    )
    say, _, y_a, _, _ = get_response_spectrum(
        acceleration_y, time_step_y, periods, damping, units, method
    )
    x_a, y_a = equalise_series(x_a, y_a)

    # TU: this is the part I m adding
    # compute vel and disp from acceleration and
    # add to the spectral acceleration time series
    from scipy.integrate import cumulative_trapezoid

    velocity_x = time_step_x * cumulative_trapezoid(acceleration_x[0:-1], initial=0.0)
    displacement_x = time_step_x * cumulative_trapezoid(velocity_x, initial=0.0)
    x_a = np.column_stack((acceleration_x[0:-1], velocity_x, displacement_x, x_a))
    velocity_y = time_step_y * cumulative_trapezoid(acceleration_y[0:-1], initial=0.0)
    displacement_y = time_step_y * cumulative_trapezoid(velocity_y, initial=0.0)
    y_a = np.column_stack((acceleration_y[0:-1], velocity_y, displacement_y, y_a))

    angles = np.arange(0.0, 90.0, 1.0)
    max_a_theta = np.zeros([len(angles), len(periods) + 3], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) * np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(
                np.max(np.fabs(x_a), axis=0) * np.max(np.fabs(y_a), axis=0)
            )
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(
                np.max(np.fabs(rot_x), axis=0) * np.max(np.fabs(rot_y), axis=0)
            )

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)
    return {
        "PGA": gmrotd[0],
        "PGV": gmrotd[1],
        "PGD": gmrotd[2],
        "Acceleration": gmrotd[3:],
    }


def compute_sa_mean(i, wf_NS, wf_EW, dt, periods, percentile):
    if i % 1000 == 0:
        print(i)
    res = gmrotdpp_withPG(
        wf_NS[i],
        dt,
        wf_EW[i],
        dt,
        periods,
        percentile,
        damping=0.05,
        units="cm/s/s",
        method="Nigam-Jennings",
    )
    return res["Acceleration"]


def parallel_processing_sa(wf_NS, wf_EW, dt, periods, percentile):
    indices = range(len(wf_NS[:, 0]))
    num_workers = min(cpu_count(), len(indices))
    pool = Pool(processes=10)

    # Wrap indices with tqdm for progress tracking
    results = list(
        tqdm(
            pool.starmap(
                compute_sa_mean, [(i, wf_NS, wf_EW, dt, periods, percentile) for i in indices]
            ),
            total=len(indices),
        )
    )

    pool.close()
    pool.join()

    return results


# Example usage
if __name__ == "__main__":
    mag = 6.2454
    rupture_aratio = 1.5
    strike = 45
    dip = 50
    rake = 0
    lon = 9.1500
    lat = 45.1833
    depth = 10
    Vs30 = 760
    hypocenter = [lon, lat, depth]
    imts = ["PGA", "PGV", "SA(0.3)", "SA(0.1)"]
    gmpes = ["BooreEtAl2014", "Kanno2006Shallow", "MorikawaFujiwara2013Crustal"]

    gms, jb_distances = calculate_gmfs(
        mag, rupture_aratio, strike, dip, rake, hypocenter, imts, Vs30, gmpes
    )
    print(gms.shape)
