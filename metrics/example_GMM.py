from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.source.characteristic import CharacteristicFaultSource
from openquake.hazardlib.mfd import ArbitraryMFD
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.valid import gsim
import numpy as np
import matplotlib.pyplot as plt

def calculate_gmfs(mag, rupture_aratio, strike, dip, rake, hypo, imts, gmps):
    
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
        name='rup',
        tectonic_region_type='Active Shallow Crust',
        mfd=ArbitraryMFD([mag], [0.01]),
        temporal_occurrence_model=PoissonTOM(50.0),
        surface=planar_surface,
        rake=rake
    )
    
    ruptures = [r for r in src.iter_ruptures()]
    
    jb_distances = np.linspace(1, 250, 300)
    
    rupture = ruptures[0]
    
    bottom_edge = Line([rupture.surface.bottom_left, rupture.surface.bottom_right])
    bottom_edge = bottom_edge.resample_to_num_points(3)
    mid_point = bottom_edge[1]
    mid_point.depth = 0.0
    
    locs = [mid_point.point_at(horizontal_distance=d, vertical_increment=0, azimuth=rupture.surface.strike + 90.)
            for d in jb_distances]
    
    site_collection = SiteCollection([Site(location=loc, vs30=760.0, vs30measured=True, z1pt0=40.0, z2pt5=1.0) for loc in locs])
    
    context_maker = ContextMaker('*', gmpes, {'imtls': imtls})
    ctxs = context_maker.from_srcs(src, site_collection)
    gms = context_maker.get_mean_stds(ctxs) #4 values (0=median, 1=std_total, 2=std_intra, 3=std_inter), then G=2 gsims, M=2 IMTs, 1 scenario = magnitude
    
    
    return gms, jb_distances

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
    hypocenter = [lon, lat, depth]
    imts = ['PGA', 'PGV', 'SA(0.3)', 'SA(0.1)']
    gmpes = ['BooreEtAl2014', 'Kanno2006Shallow', 'MorikawaFujiwara2013Crustal']
    
    gms = calculate_gmfs(mag, rupture_aratio, strike, dip, rake, hypocenter, imts, gmpes)
    print(gms.shape)

