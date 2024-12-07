"""
Emergency Warning Satellite Service (EWSS) Decoder

[1] Emergency Warning Satellite Service: Common Alert Message Format
    Specification Issue 1.0, 2024

[2] Quasi-Zenith Satellite System Interface Specification
    DCX Service (IS-QZSS-DCX-001), 2024

[3] Quasi-Zenith Satellite System Interface Specification
    DC Report Service (IS-QZSS-DCR-013), 2024

@author Rui Hirokawa

"""

import bitstruct as bs
from cssrlib.gnss import time2str, epoch2time, time2epoch, timeadd, \
    time2gpst, gpst2time
from enum import IntEnum
import json
import numpy as np
import pandas as pd


class MsgType(IntEnum):
    """ A1 – Message Type """
    TEST = 0
    ALERT = 1
    UPDATE = 2
    CLEAR = 3


class THazard(IntEnum):
    """ A4 – Hazard Category and Type """
    # CBRNE
    AIR_STRIKE = 0b0000001
    ATTACK_ON_IT = 0b0000010
    ATTACK_WITH_NUCLEAR = 0b0000011
    BIO_HAZARD = 0b0000100
    CHEMICAL_HAZARD = 0b0000101
    EXPLOSIVE_HAZARD = 0b0000110
    METEORITE_IMPACT = 0b0000111
    MISSILE_ATTACK = 0b0001000
    NUCLEAR_HAZARD = 0b0001001
    NUCLEAR_POWER_ACCIDENT = 0b0001010
    RADIOLOGICAL_HAZARD = 0b0001011
    SATELLITE_DEBRIS = 0b0001100
    SIREN_TEST = 0b0001101

    # ENVIRONMENT
    ACID_RAIN = 0b0001110
    AIR_POLLUTION = 0b0001111
    CONTAMINATED_WATER = 0b0010000
    GAS_LEAK = 0b0010001
    MARINE_POLLUTION = 0b0010010
    NOISE_POLLUTION = 0b0010011
    PLAGUE_OF_INSECTS = 0b0010100
    RIVER_POLLUTION = 0b0010101
    SUSPENDED_DUST = 0b0010110
    UV_RADIATION = 0b0010111

    # FIRE
    CONFLAGRATION = 0b0011000
    FIRE_BRIGADE_DEPLOY = 0b0011001
    FIRE_GASES = 0b0011010
    RISK_OF_FIRE = 0b0011110

    # GEO
    ASH_FALL = 0b0100000
    AVALANCHE_RISK = 0b0100001
    CRACK = 0b0100010
    DEBRIS_FLOW = 0b0100011
    EARTHQUAKE = 0b0100100
    GEOMAGNETIC_STORM = 0b0100101
    GLACIAL_ICE_AVALANCHE = 0b0100110
    LANDSLIDE = 0b0100111
    LAVA_FLOW = 0b0101000
    PYROCLASTIC_FLOW = 0b0101001
    SNOW_DRIFT = 0b0101010
    TIDAL_WAVE = 0b0101011
    TSUNAMI = 0b0101100
    VOLCANIC_MUD_FLOW = 0b0101101
    VOLCANO_ERUPTION = 0b0101110
    WIND_SURGE = 0b0101111

    # HEALRH
    EPOZOOTIC = 0b0110000
    FOOD_SAFETY_ALERT = 0b0110001
    HEALTH_HAZARD = 0b0110010
    PANDEMIC = 0b0110011
    PEST_INFESTRATION = 0b0110100
    RISK_OF_INFECTION = 0b0110101

    # INFRASTRUCTURE
    BUILDING_COLLAPSE = 0b0110110
    EMERGENCY_NUMBER_OUTAGE = 0b0110111
    GAS_SUPPLY_OUTAGE = 0b0111000
    OUTAGE_OF_IT = 0b0111001
    POWER_OUTAGE = 0b0111010
    RAW_SEWAGE = 0b0111011
    TELEPHONE_LINE_OUTAGE = 0b0111100

    # MET
    BLACK_ICE = 0b0111101
    COASTAL_FLOODING = 0b0111110
    COLD_WAVE = 0b0111111
    DERECHO = 0b1000000
    DROUGHT = 0b1000001
    DUST_STORM = 0b1000010
    FLOATING_ICSE = 0b1000011
    FLOOD = 0b1000100
    FOG = 0b01000101
    HAIL = 0b01000110
    HEAT_WAVE = 0b1000111
    LIGHTNING = 0b1001000
    POLLENS = 0b1001001
    RAINFALL = 0b1001010
    SNOW_STORM = 0b1001011
    SNOWFALL = 0b1001100
    STORM = 0b1001101
    THAWING = 0b1001110
    TORNADO = 0b1001111
    HURRICANE = 0b1010000
    WIND_CHILL = 0b1010001
    TYPHOON = 0b1010010

    # RESCUE
    DAM_FAILURE = 0b1010011
    DIKE_FAILURE = 0b1010100
    EXPLOSIVE_ORDNANCE_DISPOSAL = 0b1010101
    FACTORY_ACCIDENT = 0b1010110
    MINE_HAZARD = 0b1010111
    BOMB_DISCOVERY = 0b1011000
    DEMONSTRATION = 0b1011001
    HAZARDOUS_MATERIAL_ACCIDENT = 0b1011010
    LIFE_THREATENING_SITUATION = 0b1011011
    MAJOR_EVENT = 0b1011100
    MISSING_PERSON = 0b1011101
    RISK_OF_EXPLOTION = 0b1011110
    SAFETY_WARNING = 0b1011111
    UNDEFINED_FLYING_OBJECT = 0b1100000
    UNIDENTIFIED_ANIMAL = 0b1100001

    # SECURITY
    CHEMICAL_ATTACK = 0b1100010
    GUERRILLA_ATTACK = 0b1100011
    HIJACK = 0b1100100
    SHOOTING = 0b1100101
    SPECIAL_FORCES_ATTACK = 0b1100110
    TERRORISM = 0b1100111

    # TRANFPORT
    AIRCRAFT_CRASH = 0b1101000
    BRIDGE_COLLAPSE = 0b1101001
    DANGEROUS_GOODS_ACCIDENT = 0b1101010
    INLAND_WATERWAY_TRANSPORT_ACCIDENT = 0b1101011
    NAUTICAL_DISASTER = 0b1101100
    OIL_SPILL = 0b1101101
    ROAD_TRAFFIC_INCIDENT = 0b1101110
    TRAIN_ACCIDENT = 0b1110000

    # OTHER
    TEST_ALERT = 0b1110001


class ewsDec():
    """ EWS Message decoder class """

    def load_msg(self, file):
        s = {}
        with open(self.msg_path+file, "rt", encoding="utf-8") as f:
            for line in f:
                v = line.split('\t')
                key = int(v[0])
                s[key] = v[1].strip()
        return s

    def __init__(self, bdir='../data/ews/', year=0):
        self.msg_path = bdir
        self.monlevel = 0
        self.year = year

    def decode(self, msg, i):
        None

    def add_day(self, month, day, nday=0):
        t0 = epoch2time([self.year, month, day, 0, 0, 0.0])
        t1 = timeadd(t0, nday*86400.0)
        ep = time2epoch(t1)
        return ep[1], ep[2]

    def draw_ellipse(self, ax, lat, lon, LM, lm, az):
        """ draw ellipse (TBD) """
        None


class jmaDec(ewsDec):
    """ JMA DC Report decoder class """

    def __init__(self, bdir='../data/ewss/jma/', year=0):
        super().__init__(bdir=bdir, year=year)

        self.rc_m_t = {3: 'Regular', 7: 'Test'}
        self.dc_m_t = {1: 'JMA Earthquake',
                       2: 'JMA Hypocenter',
                       3: 'JMA Seismic Intencity',
                       4: 'JMA Nankai Trough Earthquake',
                       5: 'JMA Tsunami',
                       6: 'JMA North Pacific Tsunami',
                       8: 'JMA Volcano',
                       9: 'JMA Ash Fall',
                       10: 'JMA Weather',
                       11: 'JMA Flood',
                       12: 'JMA Typhoon',
                       14: 'JMA Marine'}

        self.itype_t = {0: 'Issue', 2: 'Cancellation'}

        self.co_t = self.load_msg('tab4_1_2_6.txt')

        # earthquake
        self.ep_t = self.load_msg('tab4_1_2_7.txt')
        self.sil_t = self.load_msg('tab4_1_2_8.txt')
        self.siu_t = self.load_msg('tab4_1_2_9.txt')
        self.pre_t = self.load_msg('tab4_1_2_10.txt')
        self.longm_t = self.load_msg('tab4_1_2_11.txt')

        # intensity
        self.mag_i_t = self.load_msg('tab4_1_2_15.txt')
        self.pre_i_t = self.load_msg('tab4_1_2_16.txt')

        self.isc_t = self.load_msg('tab4_1_2_19.txt')

        # Tsunami
        self.dw_w_t = self.load_msg('tab4_1_2_22.txt')
        self.th_w_t = self.load_msg('tab4_1_2_23.txt')
        self.reg_w_t = self.load_msg('tab4_1_2_24.txt')

        # Volcano
        self.dw_v_t = self.load_msg('tab4_1_2_31.txt')
        self.vo_t = self.load_msg('tab4_1_2_32.txt')
        self.reg_v_t = self.load_msg('tab4_1_2_33.txt')

        self.dw_t = self.load_msg('tab4_1_2_36.txt')

        # Weather
        self.ww_t = self.load_msg('tab4_1_2_40.txt')
        self.reg_ww_t = self.load_msg('tab4_1_2_41.txt')

        # Flood
        self.lv_t = self.load_msg('tab4_1_2_44.txt')
        self.reg_fl_t = self.load_msg('tab4_1_2_45.txt')

        # for Marine
        self.dw_m_t = self.load_msg('tab4_1_2_52.txt')
        self.reg_m_t = self.load_msg('tab4_1_2_53.txt')

        self.pidx = 0
        self.msg = bytearray(512)

        self.vn = -1

    def decode_jma_earthquake(self, msg, i):
        """ JMA Earthquake """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _ = bs.unpack_from('u2u4', msg, i)
        i += 6

        lgL1, lgU1 = bs.unpack_from('u3u3', msg, i)
        i += 6

        if lgL1 == lgU1:
            long_ = f"{self.longm_t[lgL1]}"
        else:
            long_ = f"{self.longm_t[lgL1]}{self.longm_t[lgU1]}"

        co_ = []
        for k in range(3):
            co_.append(bs.unpack_from('u9', msg, i)[0])
            i += 9
        d1, h1, m1 = bs.unpack_from('u5u5u6', msg, i)
        i += 16

        de, ma = bs.unpack_from('u9u7', msg, i)
        i += 16
        ma *= 0.1

        ep, L1, U1 = bs.unpack_from('u10u4u4', msg, i)
        i += 18

        if L1 == U1:
            int_ = f"{self.sil_t[L1]}"
        else:
            int_ = f"{self.sil_t[L1]}{self.siu_t[U1]}"

        reg_ = []
        for k in range(80):
            j = bs.unpack_from('u1', msg, i)[0]
            if j > 0:
                reg_.append(k)
            i += 1
        i += 4
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Earthquake ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")

            for k in range(3):
                if co_[k] > 0:
                    print(f"{self.co_t[co_[k]]}")

            print(f"occurance time: {month}/{d1} {h1:2d}:{m1:02d}")
            print(f"depth={de}km mag={ma:.1f}")
            print(f"{long_}")
            print(f"epicenter:{self.ep_t[ep]} {int_}")

            s_ = ""
            for reg in reg_:
                s_ += self.pre_t[reg]+" "
            print(s_)

        return i

    def decode_jma_hypocenter(self, msg, i):
        """ JMA Epicenter """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _ = bs.unpack_from('u2u10', msg, i)
        i += 12

        co_ = []
        for k in range(3):
            co_.append(bs.unpack_from('u9', msg, i)[0])
            i += 9
        d1, h1, m1 = bs.unpack_from('u5u5u6', msg, i)
        i += 16
        de, ma, ep = bs.unpack_from('u9u7u10', msg, i)
        i += 26
        ma *= 0.1

        s, latd, latm, lats = bs.unpack_from('u1u7u6u6', msg, i)
        i += 20

        lat = latd+latm/60.0+lats/3600.0
        if s == 1:
            lat = -lat

        s, lond, lonm, lons = bs.unpack_from('u1u8u6u6', msg, i)
        i += 21

        lon = lond+lonm/60.0+lons/3600.0
        if s == 1:
            lon = -lon

        i += 51
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Epicenter ({self.itype_t[itype]})" +
                  f" {month:2d}/{day:02d} {hour:2d}:{minute:02d}")

            for k in range(3):
                if co_[k] > 0:
                    print(f"{self.co_t[co_[k]]}")

            print(f"occurance time: {month}/{d1} {h1:2d}:{m1:02d}")
            print(f"depth={de}km mag={ma:.1f}")
            print(f"epicenter:{self.ep_t[ep]} lat {lat:.4f} lon {lon:.4f}")

        return i

    def decode_jma_seismic_intencity(self, msg, i):
        """ JMA Seismic Intensity """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _ = bs.unpack_from('u2u10', msg, i)
        i += 12

        d1, h1, m1 = bs.unpack_from('u5u5u6', msg, i)
        i += 16

        es_ = []
        reg_ = []
        for k in range(16):
            es, reg = bs.unpack_from('u3u6', msg, i)
            i += 9
            if es == 0 and reg == 0:
                continue
            es_.append(es)
            reg_.append(reg)

        i += 1

        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Intensity ({self.itype_t[itype]})" +
                  f" {month:2d}/{day:02d} {hour:2d}:{minute:02d}")

            print(f"occurance time: {month}/{d1} {h1:2d}:{m1:02d}")
            for k, reg in enumerate(reg_):
                print(f"{self.mag_i_t[es_[k]]} {self.pre_i_t[reg]}")

        return i

    def decode_jma_nankai_earthquake(self, msg, i):
        """ JMA Nankai Earthquake """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _, is_ = bs.unpack_from('u2u10u4', msg, i)
        i += 16

        txt = bytearray(18)
        for k in range(18):
            txt[k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

        pn, pm = bs.unpack_from('u6u6', msg, i)
        i += 13

        self.pidx |= 1 << (pn-1)
        self.msg[(pn-1)*18:pn*18] = txt

        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Nankai Trough Earthquake ({self.itype_t[itype]}) " +
                  f"{self.isc_t[is_]} " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d} {pn}/{pm}")
            if self.pidx == (1 << pm)-1:
                print(self.msg.decode('utf-8'))
        return i

    def decode_jma_tsunami(self, msg, i):
        """ JMA Tsunami """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _ = bs.unpack_from('u2u10', msg, i)
        i += 12

        if self.monlevel > 0:
            print(f"Tsunami ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")

        for k in range(3):
            co = bs.unpack_from('u9', msg, i)[0]
            i += 9

            if self.monlevel > 0:
                if co > 0 and co in self.co_t:
                    print(f"{self.co_t[co]}")

        dw = bs.unpack_from('u4', msg, i)[0]
        i += 4

        if self.monlevel > 0:
            if dw > 0 and dw in self.dw_t:
                print(f"{self.dw_w_t[dw]}")

        for k in range(5):
            d, h, m = bs.unpack_from('u1u5u6', msg, i)
            i += 12
            th, reg = bs.unpack_from('u4u10', msg, i)
            i += 14

            if d == 0 and h == 0 and m == 0 and th == 0 and reg == 0:
                continue

            if self.monlevel > 0:
                print(f"{d} {h:2d}:{m:02d} " +
                      f"{self.th_w_t[th]} {self.reg_w_t[reg]}")

        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        return i

    def decode_jma_nw_pacific_tsunami(self, msg, i):
        """ JMA North Pacific Tsunami """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _, tp = bs.unpack_from('u2u10u3', msg, i)
        i += 15

        for k in range(5):
            d, h, m = bs.unpack_from('u1u5u6', msg, i)
            i += 12
            th, reg = bs.unpack_from('u9u7', msg, i)
            i += 16

        i += 18
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(
                f"NP Tsunami ({itype}) {month:2d}/{day:02d} " +
                f"{hour:2d}:{minute:02d}")

        return i

    def decode_jma_volcano(self, msg, i):
        """ JMA Volcano """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _, du = bs.unpack_from('u2u7u3', msg, i)
        i += 12

        d1, h1, m1 = bs.unpack_from('u5u5u6', msg, i)
        i += 16
        dw, vo = bs.unpack_from('u7u12', msg, i)
        i += 19

        reg_ = []
        for k in range(5):
            reg = bs.unpack_from('u23', msg, i)[0]
            if reg > 0:
                reg_.append(reg)
            i += 23
        i += 11
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Volcano ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")
            print(f"du={du} activity time: {month}/{d1} {h1:2d}:{m1:02d}")
            print(f"{self.dw_v_t[dw]} {self.vo_t[vo]}")

            for reg in reg_:
                print(self.reg_v_t[reg])

        return i

    def decode_jma_ash_fall(self, msg, i):
        """ JMA Ash Fall """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype = bs.unpack_from('u2', msg, i)[0]
        i += 12

        d1, h1, m1, dw1, vo = bs.unpack_from('u5u5u6u2u12', msg, i)
        i += 30

        vo_ = self.vo_t[vo]

        self.dw1_t = {1: "Ash Fall Forecast (Preliminary)",
                      2: "Ash Fall Forecast (Detailed)"}

        if self.monlevel > 0:
            print(f"Ash Fall ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")
            print(f" {month}/{d1} {h1:2d}:{m1:02d} {self.dw1_t[dw1]} {vo_}")

        for k in range(4):
            ho, dw2, reg = bs.unpack_from('u3u3u23', msg, i)
            i += 29
            if reg > 0:
                reg_ = self.reg_v_t[reg]
                dw_ = self.dw_t[dw2]
                if self.monlevel > 0:
                    print(f"{ho}h {dw_} {reg_}")

        i += 15
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12

        return i

    def decode_jma_weather(self, msg, i):
        """ JMA Weather """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _, ar = bs.unpack_from('u2u10u3', msg, i)
        i += 15

        self.ar_t = {1: "issue", 2: "cansellation"}

        if self.monlevel > 0:
            print(f"Weather ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d} " +
                  f"{self.ar_t[ar]}")

        for k in range(6):
            ww, pl = bs.unpack_from('u5u19', msg, i)
            i += 24
            if ww == 0 or pl == 0:
                continue
            if self.monlevel > 0:
                print(f"{self.ww_t[ww]} {self.reg_ww_t[pl]}")

        i += 14  # spare 2
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12
        return i

    def decode_jma_flood(self, msg, i):
        """ JMA Flood """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _ = bs.unpack_from('u2u10', msg, i)
        i += 12

        if self.monlevel > 0:
            print(f"Flood ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")

        for k in range(3):
            lv, reg = bs.unpack_from('u4u40', msg, i)
            i += 44

            if lv == 0 and reg == 0:
                continue
            if self.monlevel > 0:
                print(f"{self.lv_t[lv]} {self.reg_fl_t[reg]}")

        i += 29
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12
        return i

    def decode_jma_typhoon(self, msg, i):
        """ JMA Tyhoon """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype, _, d1, h1, m1, dt = bs.unpack_from('u2u10u5u5u6u3', msg, i)
        i += 31+8

        self.dt_tp_t = {1: 'Analysis', 2: 'Estimate', 3: 'Forecast'}

        du, tn, sr, ic = bs.unpack_from('u7u7u4u4', msg, i)
        i += 22

        s, latd, latm, lats = bs.unpack_from('u1u7u6u6', msg, i)
        i += 20
        lat = latd+latm/60.0+lats/3600.0
        if s == 1:
            lat = -lat

        s, lond, lonm, lons = bs.unpack_from('u1u8u6u6', msg, i)
        i += 21
        lon = lond+lonm/60.0+lons/3600.0
        if s == 1:
            lon = -lon

        pr, w1, w2 = bs.unpack_from('u11u7u7', msg, i)
        i += 71

        if self.monlevel > 0:
            print(f"Typhoon ({self.itype_t[itype]}) #{tn} ({sr}/{ic}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d} " +
                  f"{month}/{d1} {h1:2d}:{m1:02d} {self.dt_tp_t[dt]} {du}h " +
                  f"lat={lat:.4f} lon={lon:.4f} pr={pr} w1={w1} w2={w2}")

        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12
        return i

    def decode_jma_marine(self, msg, i):
        """ JMA Marine """
        month, day, hour, minute = bs.unpack_from('u4u5u5u6', msg, i)
        i += 20
        itype = bs.unpack_from('u2', msg, i)[0]
        i += 12

        if self.monlevel > 0:
            print(f"Manine ({self.itype_t[itype]}) " +
                  f"{month:2d}/{day:02d} {hour:2d}:{minute:02d}")

        for k in range(8):
            dw, reg = bs.unpack_from('u5u14', msg, i)
            i += 19

            if dw == 0 and reg == 0:
                continue

            if self.monlevel > 0:
                print(f"{self.dw_m_t[dw]} {self.reg_m_t[reg]}")

        i += 9
        self.vn = bs.unpack_from('u6', msg, i)[0]
        i += 12
        return i

    def decode(self, msg, i):
        """ decode DC-report messages """
        self.rc, self.dc = bs.unpack_from('u3u4', msg, i)
        i += 7

        if self.monlevel > 0:
            print(f"[DCR] {self.rc_m_t[self.rc]} {self.dc_m_t[self.dc]}")

        if self.dc == 1:
            self.decode_jma_earthquake(msg, i)
        elif self.dc == 2:
            self.decode_jma_hypocenter(msg, i)
        elif self.dc == 3:
            self.decode_jma_seismic_intencity(msg, i)
        elif self.dc == 4:
            self.decode_jma_nankai_earthquake(msg, i)
        elif self.dc == 5:
            self.decode_jma_tsunami(msg, i)
        elif self.dc == 6:
            self.decode_jma_nw_pacific_tsunami(msg, i)
        elif self.dc == 8:
            self.decode_jma_volcano(msg, i)
        elif self.dc == 9:
            self.decode_jma_ash_fall(msg, i)
        elif self.dc == 10:
            self.decode_jma_weather(msg, i)
        elif self.dc == 11:
            self.decode_jma_flood(msg, i)
        elif self.dc == 12:
            self.decode_jma_typhoon(msg, i)
        elif self.dc == 14:
            self.decode_jma_marine(msg, i)


class camfDec(ewsDec):
    """ Common Alert Message Format (CAMF) Decoder class """

    def __init__(self, bdir='../data/ews/camf/', year=0):
        super().__init__(bdir=bdir, year=year)
        self.bdir = bdir

        self.mt_t = ('Test', 'Alert', 'Update', 'All Clear')
        self.hazard_t = self.load_msg('hazard.txt')

        # JIS X0401
        with open(bdir+'jisx0401-en.json', 'r',
                  encoding='utf-8') as fh:
            self.pref_t = json.load(fh)

        # JIS X0402
        self.mc_t = pd.read_csv(bdir+'000323625.csv', encoding='sjis')

        self.bdir = bdir
        self.city_t = None

        self.list_a = self.load_msg('list_a.txt')
        self.list_b = self.load_msg('list_b.txt')
        self.list_c = self.load_msg('list_c.txt')

        self.region_t = self.load_msg('region.txt')

        self.severity_t = ('unknown', 'moderate', 'severe', 'extreme')

        self.duration_t = ('unknown', 'duration<6h', '6h<duration<12h',
                           '12h<duration<24h')

        self.r_t = [0.216, 0.292, 0.395, 0.535, 0.723, 0.978, 1.322, 1.788,
                    2.418, 3.269, 4.421, 5.979, 8.085, 10.933, 14.784, 19.992,
                    27.035, 36.559, 49.439, 66.855, 90.407, 122.255, 165.324,
                    223.564, 302.322, 408.824, 552.846, 747.603, 1010.970,
                    1367.116, 1848.727, 2500.0]

        self.d1_t = ('1.0-1.9', '2.0-2.9', '3.0-3.9', '4.0-4.9', '5.0-5.9',
                     '6.0-6.9', '7.0-7.9', '8.0-8.9', '9.0-')
        # D2 – Seismic Coefficient
        self.d2_t = ('2', '3', '4', '5 Weak', '5 Strong',
                     '6 Weak', '6 Strong', '7')
        # D4 – Vector Length between Centre of Main Ellipse and Epicentre
        self.d4_t = (0.25, 0.5, 0.75, 1, 2, 3, 5, 10,
                     20, 30, 40, 50, 70, 100, 150, 200)
        # D5 – Wave Height
        self.d5_t = ('H<=0.5', '0.5<H<=1.0', '1.0<H<=1.5',
                     '1.5<H<=2.0', '2.0<H<=3.0', '3.0<H<=5.0',
                     '5.0<H<=10.0', 'H>10.0')

        self.d6_t = ('T<=-30', '-30<T<=-25', '-25<T<=-20',
                     '-20<T<=-15', '-15<T<=-10', '-10<T<=-5', '-5<T<=0',
                     '0<T<=5', '5<T<=10', '10<T<=15', '15<T<=20',
                     '20<T<=25', '25<T<=30', '30<T<=35', '35<T<=45', 'T>45')

        self.d7_t = ('Category 1/5 hurricane', 'Category 2/5 hurricane',
                     'Category 3/5 hurricane', 'Category 4/5 hurricane',
                     'Category 5/5 hurricane')
        self.d8_t = ('0<v<1 km/h', '1<v<6 km/h', '6<v<11 km/h', '12<v<19 km/h',
                     '20<v<30 km/h', '31<v<39 km/h', '40<v<50 km/h',
                     '51<v<61 km/h', '62<v<74 km/h', '75<v<88 km/h',
                     '89<v<102 km/h', '103<v<117 km/h', 'v>118 km/h')
        self.d9_t = ('p<=2.5', '2.5<p<=7.5', '7.5<p<=10', '10<p<=20',
                     '20<p<=30', '30<p<=50', '50<p<=80', '80<p')
        # D10 – Damage Category
        self.d10_t = (
            'Category 1 - Very dangerous winds will produce some damage. Scale 1 and Intensity 1',
            'Category 2 - Extremely dangerous winds will cause extensive damage. Scale 1 and Intensity 2',
            'Category 3 - Devastating damage will occur. Scale 1 and Intensity 3',
            'Category 4 - Catastrophic damage will occur. Scale 2 and Intensity 1',
            'Category 5 - Catastrophic damage will occur. Scale 2 and Intensity 2',
            'Category 5 - Catastrophic damage will occur. Scale 3 and Intensity 3'
        )
        # D11 – Tornado Probability
        self.d11_t = ('Non-Threatening', 'Very Low', 'Low', 'Moderate', 'High',
                      'Extreme')
        # D12 – Hail Scale
        self.d12_t = ('H0 Hard hail', 'H1 Potentially damaging',
                      'H2 Significant', 'H3 Severe', 'H4 Severe',
                      'H5 Destructive', 'H6 Destructive', 'H7 Destructive',
                      'H8 Destructive', 'H9 Super Hailstorms',
                      'H10 Super Hailstorms')
        # D13 – Visibility
        self.d13_t = ('Dense fog: visibility < 20 m',
                      'Thick fog: 20 m < visibility < 200 m',
                      'Moderate fog: 200 m < visibility < 500 m',
                      'Light fog: 500 m < visibility < 1000 m',
                      'Thin fog: 1 km < visibility < 2 km',
                      'Haze: 2 km < visibility < 4 km',
                      'Light haze: 4 km < visibility < 10 km',
                      'Clear: 10 km < visibility < 20 km',
                      'Very clear: 20 km < visibility < 50 km',
                      'Exceptionally clear: visibility > 50 km')
        # D14 - Snow depth: d14.txt
        self.d14_t = self.load_msg('d14.txt')

        # D15 – Flood Severity
        self.d15_t = ('Minor Flooding', 'Moderate Flooding', 'Major Flooding',
                      'Record Flooding')
        # D16 – Lightning Intensity
        self.d16_t = ('LAL 1 - No thunderstorms',
                      'LAL 2 - Isolated thunderstorms',
                      'LAL 3 - Widely scattered thunderstorms',
                      'LAL 4 - Scattered thunderstorms',
                      'LAL 5 - Numerous thunderstorms',
                      'LAL 6 - Dry lightning')
        # D17 – Fog Level
        self.d17_t = (
            'Level 1 of 5: Slight fog or Mist',
            'Level 2 of 5: Slight fog',
            'Level 3 of 5: Moderate fog',
            'Level 4 of 5: Moderate fog',
            'Level 5 of 5: Thick fog')
        # D18 – Drought Level
        self.d18_t = ('D1 – Moderate Drought – PDSI = -2.0 to -2.9',
                      'D2 – Severe Drought – PDSI = -3.0 to -3.9',
                      'D3 – Extreme Drought – PDSI = -4.0 to -4.9',
                      'D4 – Exceptional Drought – PDSI = -5.0 or less')
        # D19 – Avalanche Warning Level
        self.d19_t = ('Low', 'Moderate', 'Considerable', 'High', 'Very high')
        # D20 – Ash Fall Amount and Impact
        self.d20_t = ('Less than 1 mm ash thickness',
                      '1-5 mm ash thickness',
                      '5-100 mm ash thickness',
                      '100-300 mm ash thickness',
                      '> 300 mm ash thickness')
        # D21 - Geomagnetic Scale
        self.d21_t = ('G1 - Minor', 'G2 - Moderate', 'G3 - Strong',
                      'G4 - Severe', 'G5 - Extreme')
        # D22 – Terrorism Threat Level
        self.d22_t = ('Very low threat level', 'Low threat level',
                      'Medium threat level', 'High threat level',
                      'Critical threat level')
        # D23 – Fire Risk Level
        self.d23_t = ('Danger level 1/5 (low or none danger)',
                      'Danger level 2/5 (moderate danger)',
                      'Danger level 3/5 (considerable danger)',
                      'Danger level 4/5 (high danger)',
                      'Danger level 5/5 (very high danger)')
        # D24 – Water Quality
        self.d24_t = ('Excellent water quality', 'Good water quality',
                      'Poor water quality', 'Very poor water quality',
                      'Suitable for drinking purposes',
                      'Unsuitable for drinking purpose')
        # D25 – UV Index
        self.d25_t = ('Index 0-2 Low', 'Index 3 Moderate', 'Index 4 Moderate',
                      'Index 5 High', 'Index 6 High', 'Index 7 High',
                      'Index 8 Very high', 'Index 9 Very high',
                      'Index 10 Extreme', 'Index 11 Extreme')
        # D26 – Number of Cases per 100 000 Inhabitants
        self.d26_t = ('0-9', '10-20', '21-50', '51-70', '71-100', '101-125',
                      '126-150', '151-175', '176-200', '201-250', '251-300',
                      '301-350', '351-400', '401-450', '451-500', '501-750',
                      '751-1000', '>1000', '>2000', '>3000', '>5000')
        # D27 – Noise Range
        self.d27_t = ('40<dB≤45', '45<dB≤50', '50<dB≤60', '60<dB≤70',
                      '70<dB≤80 (loud)', '80<dB≤90 (very loud)',
                      '90<dB≤100 (very loud)',
                      '100<dB≤110 (very loud)',
                      '110<dB≤120 (extremely loud)',
                      '120<dB≤130 (extremely loud)',
                      '130<dB≤140 (threshold of pain)',
                      'dB>140 (pain)')
        # D28 - Air Quality Index
        self.d28_t = ('0-50 Good', '51-100 Moderate',
                      '101-150 Unhealthy for sensitive groups',
                      '151-200 Unhealthy',
                      '201-300 Very unhealthy',
                      '301-500 Hazardous')
        # D29 – Outage Estimated Duration
        self.d29_t = ('0 < duration < 30 min',
                      '30 min ≤ duration < 45 min',
                      '45 min ≤ duration < 1 h',
                      '1 h ≤ duration < 1h 30min',
                      '1h 30min ≤ duration < 2 h',
                      '2 h ≤ duration < 3 h',
                      '3 h ≤ duration < 4 h',
                      '4 h ≤ duration < 5 h',
                      '5 h ≤ duration < 10 h',
                      '10 h ≤ duration < 24 h',
                      '24 h ≤ duration < 2 days',
                      '2 days ≤ duration < 7 days',
                      '7 days ≤ duration', 'Unknown')
        # D30 – Nuclear Event Scale
        self.d30_t = ('Unknown', 'Level 0 Deviation', 'Level 1 Anomaly',
                      'Level 2 Incident', 'Level 3 Serious incident',
                      'Level 4 Accident with local consequences',
                      'Level 5 Accident with wider consequences',
                      'Level 6 Serious accident',
                      'Level 7 Major accident')
        # D31 – Chemical Hazard Type
        self.d31_t = ('Explosives', 'Flammable gases',
                      'Flammable aerosols and aerosols',
                      'Oxidizing gases', 'Gases under pressure',
                      'Flammable liquids', 'Flammable solids',
                      'Self-reactive substance/mixture',
                      'Pyrophoric liquids', 'Pyrophoric solids',
                      'Self-heating substance/mixture',
                      'Water-reactive',
                      'Oxidising liquids',
                      'Oxidising solids',
                      'Organic peroxides',
                      'Corrosive to metals')
        # D32 – Biohazard Level
        self.d32_t = ('Biohazard Level 1/4', 'Biohazard Level 2/4',
                      'Biohazard Level 3/4', 'Biohazard Level 4/4')
        # D33 - Biohazard Type
        self.d33_t = ('Biological agents', 'Biotoxins',
                      'Blood and blood products', 'Environmental specimens')
        # D34 – Explosive Hazard Type
        self.d34_t = ('PE1 - Mass explosion hazard',
                      'PE2 - Serious projectile hazard',
                      'PE3 - Fire and a minor blast/projection hazard',
                      'PE4 - Fire or slight explosion hazard')
        # D35 – Infection Type => d35.txt
        self.d35_t = self.load_msg('d35.txt')

        # D36 – Typhoon Categories
        self.d36_t = ('Scale 1 and Intensity 1',
                      'Scale 1 and Intensity 2',
                      'Scale 1 and Intensity 3',
                      'Scale 2 and Intensity 1',
                      'Scale 2 and Intensity 2')

        # data
        self.sel = 0

        self.lat = 0.0
        self.lon = 0.0
        self.LM = self.Lm = self.az = 0.0
        self.dlat = 0.0
        self.dlon = 0.0

        self.dist = 0.0
        self.LMs = 0.0
        self.Lms = 0.0
        self.theta = 0.0

        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.d4 = None
        self.d5 = None
        self.d6 = None
        self.d7 = None
        self.d8 = None
        self.d9 = None
        self.d10 = None
        self.d11 = None
        self.d12 = None
        self.d13 = None
        self.d14 = None
        self.d15 = None
        self.d16 = None
        self.d17 = None
        self.d18 = None
        self.d19 = None
        self.d20 = None
        self.d21 = None
        self.d22 = None
        self.d23 = None
        self.d24 = None
        self.d25 = None
        self.d26 = None
        self.d27 = None
        self.d28 = None
        self.d29 = None
        self.d30 = None
        self.d31 = None
        self.d32 = None
        self.d33 = None
        self.d34 = None
        self.d35 = None

        self.dow_t = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                      'Saturday', 'Sunday')

        self.inst_t = ('(not specified)', 'stay',
                       'move to/toward', 'keep away from')

        self.guid_t = {
            0: '(not specified)',
            1: 'Under/inside a solid structure',
            2: '3rd floor or higher',
            3: 'Underground',
            4: 'Mountain',
            5: 'Water area',
            6: 'Building where chemicals are handled, such as a factory',
            7: 'Cliffs and areas at risk of collapse',
            127: 'Take the best immediate action to save your life',
            255: 'Take the best immediate action to save your life'
        }

        # inst = 0
        self.guid_la_t = {
            0: '(not specified)',
            1: 'Take the best immediate action to save your life.',
            127: 'Take the best immediate action to save your life.',
            128: 'Missile launched.',
            129: 'Missile passed.',
            130: 'It is believed that the previous missile has dropped in the sea.',
            131: 'It is believed that the previous missile will not come to Japan.',
            132: 'Take shelter immediately.',
            133: 'The previous missile has been intercepted and destroyed.',
            134: 'Missile dropped',
            135: 'It is believed that the previous missile will not drop in Japan.',
            136: 'This is a test message for J-Alert.',
            255: 'Take immediate action to save your life.'
        }

        # inst = 1
        self.guid_ja1_t = {
            0: 'Stay.',
            1: 'Stay. Under/inside a solid structure.',
            2: 'Stay. 3rd floor or higher.',
            3: 'Stay. Underground.',
            4: 'Stay. Mountain.',
            5: 'Stay. Water area.',
            6: 'Stay. Building where chemicals are handled, such as a factory.',
            7: 'Stay. Cliffs and areas at risk of collapse.'
        }

        # inst = 2
        self.guid_ja2_t = {
            0: 'Keep away from',
            1: 'Keep away from Under/inside a solid structure.',
            2: 'Keep away from 3rd floor or higher.',
            3: 'Keep away from Underground.',
            4: 'Keep away from Mountain.',
            5: 'Keep away from Water area.',
            6: 'Keep away from Building where chemicals are handled, such as a factory.',
            7: 'Keep away from Cliffs and areas at risk of collapse.',

        }

    def decode_ext(self, msg, i):
        """ decode DCX Extended Message (74bits) """

        if self.sel == 0:  # information from organizations outside Japan
            i += 68  # EX11 TBD
            vn = bs.unpack_from('u6', msg, i)[0]
            i += 6
            return i

        # for Japan
        if self.pid == 1:  # L-Alert
            if self.city_t is None:
                with open(self.bdir+'City_list.json', 'r',
                          encoding='utf-8') as fh:
                    self.city_t = json.load(fh)

            # EX1 target area code
            # EX2 Evacuate Direction Type
            # EX3 Additional Ellipse Centre Latitude
            # EX4 Additional Ellipse Centre Longitude
            # EX5 Additional Ellipse Semi-Major Axis
            # EX6 Additional Ellipse Semi-Minor Axis
            # EX7 Additional Ellipse Azimuth

            self.ed_t = ('Leave the additional target area range.',
                         'Head to the additional target area range.')

            ex1, ex2, ex3, ex4 = bs.unpack_from('u16u1u17u17', msg, i)
            i += 51

            ex5, ex6, ex7, vn = bs.unpack_from('u5u5u7u6', msg, i)
            i += 23

            pref_code = '{:02d}'.format(ex1//1000)
            city_code = '{:03d}'.format(ex1 % 1000)

            if ex1 == 0:
                return i

            name = ''
            for city in self.city_t['cities']:
                if city['pref_code'] == pref_code and \
                        city['city_code'] == city_code:
                    name = city['name']
                    break

            if len(name) == 0:
                return i

            ed = self.ed_t[ex2]

            # ex3-ex7 not used in L-Alert
            lat = -90+180/131071*ex3
            lon = 45+180/131071*ex4

            LM = self.r_t[ex5]
            Lm = self.r_t[ex6]
            az = -90+180/128*ex7

            if self.monlevel > 0:
                print(f"[DCX-ext] pid={self.pid} {time2str(self.time)} {name} " +
                      f"{ed} {lat} {lon} {LM} {Lm} {az} {vn}")

        elif self.pid == 2 or self.pid == 3:  # J-Alert
            # EX8 Target Area Code List Type
            # EX9 Target Area Code List

            ex8 = bs.unpack_from('u1', msg, i)[0]
            i += 1
            ts = time2str(self.time)

            if ex8 == 0:  # prefecture code (JISX0401)
                ex9 = bs.unpack_from('u47', msg, i)[0]
                i += 64
                lst = []
                for k in range(47):
                    if (ex9 >> (46-k)) & 0x1:
                        key = f"{k+1:02d}"
                        lst.append(self.pref_t[key])
                if len(lst) == 47:
                    s = "whole Japan"
                else:
                    s = ""
                    for key_ in lst:
                        s += key_ + " "

                # ex9 target are code list
                if self.monlevel > 0:
                    print(f"[DCX-ext] {self.pid} {ex8} {s}")
            else:  # municipality code (JISX0402)
                s = ''
                for k in range(4):
                    mc = bs.unpack_from('u16', msg, i)[0]
                    i += 16
                    if mc > 0:
                        j = np.where(self.mc_t['tiiki-code'] == mc)[0][0]
                        si = self.mc_t['ken-name'][j]
                        if pd.isna(self.mc_t['sityouson-name1'][j]) == False:
                            si += self.mc_t['sityouson-name1'][j]
                        if pd.isna(self.mc_t['sityouson-name2'][j]) == False:
                            si += self.mc_t['sityouson-name2'][j]
                        if pd.isna(self.mc_t['sityouson-name3'][j]) == False:
                            si += self.mc_t['sityouson-name3'][j]
                        s += si + " "
                if self.monlevel > 0:
                    print(f"[DCX-ext] {self.pid} {ex8} {s}")

            ex10, vn = bs.unpack_from('u3u6', msg, i)
            i += 9

        return i

    def decode(self, msg, i):
        """ decode CAMF message (122bits) """
        # 3.1 Message Identifier (A1,A2,A3)
        mt, self.regi, pid = bs.unpack_from('u2u9u5', msg, i)
        i += 16

        # mt: 0:Test,1:Alert,2:Update,3:All clear
        self.mt = self.mt_t[mt]
        # reg: Tab. 4.2-5 (ISO 3166)
        self.region = self.region_t[self.regi]
        # pid: Tab. 4.2-6
        self.pid = pid  # Provider identifier (A3)

        # if region is Japan:
        #  pid=1 (FMMC) -> L-Alert
        #  pid=2 (FDMA) or pid=3 (Related Ministries) -> J-Alert
        #  pid=4 (Municipality) -> municipality-transmitted info

        # 3.2 Hazard
        hcat, sev = bs.unpack_from('u7u2', msg, i)
        i += 9

        # hcat: Tab. 4.2-7
        if hcat not in self.hazard_t.keys():
            i += 97
            if hcat > 0:
                print(f"category not found: pid={pid} cat={hcat} sev={sev}")
            return i

        self.hazard = self.hazard_t[hcat]
        # sev: 0:unknown,1:moderate,2:severe,3:extreme
        self.severity = self.severity_t[sev]

        # 3.3 Hazard Chronology (beginning of hazard)
        wn, tow, dur = bs.unpack_from('u1u14u2', msg, i)
        i += 17

        # wn: 0:current week, 1: next week
        # tow: 1:mon 00:00, 2:mon 00:01, 3:mon 00:02
        # .. sun 23:59
        dow_ = (tow-1) // 1440
        min_ = (tow-1) % 1440
        hour_ = min_ // 60
        min_ -= hour_*60

        week, tow = time2gpst(self.time)
        week += wn
        tow_ = dow_*86400+hour_*3600+min_*60

        # harzrd onset (UTC)
        self.th = gpst2time(week, tow_)

        self.epoch = time2epoch(self.th)

        # self.epoch = [wn, dow_, hour_, min_]

        # duration (A8) : 0:unknown,1:<6h,2:6h<=d<12h,3:12h<=d<24h
        self.duration = self.duration_t[dur]

        # 3.4 Alert Identifier

        # 3.5 Guidance to react (A9, A10)
        self.sel, self.ver = bs.unpack_from('u1u3', msg, i)
        i += 4

        if self.sel == 0:  # international (EU)
            ia, ib = bs.unpack_from('u5u5', msg, i)
            # List A: General required action
            # List B: Monitoring + Required action
            self.gmsg_a = self.list_a[ia]
            self.gmsg_b = self.list_b[ib]
        else:  # 1:country/region (Japan..)
            if self.regi == 0b001101111:  # Japan
                inst, info = bs.unpack_from('u2u8', msg, i)
                self.inst = self.inst_t[inst]
                if pid == 2 or pid == 3:  # L-Alert
                    self.guid = self.guid_la_t[info]
                    s = self.guid
                else:
                    self.guid = self.guid_t[info]
                    s = '{:s} {:s}'.format(self.inst, self.guid)
            else:
                print(f"not supported region: {self.regi}")
                i += 76
                return i

        i += 10

        # sel: 0:international, 1:country/region (Japan..)
        # guid:
        # Japan (sel=1):
        #  instruction (2) : 0:not specified, 1:stay, 2:move, 3:keep away
        #  info (8): Tab. 4.2-14
        # EU (sel=0):
        #  list-A(5): Tab. 4.2-14
        #  list-B(5): Tab. 4.2-15
        self.ver += 1

        # 3.6 Target Area
        lati, loni, smai, smii, azi = bs.unpack_from('u16u17u5u5u6', msg, i)
        i += 49

        self.lat = -90.0+180.0/65535*lati
        self.lon = -180.0+360.0/131071*loni

        self.LM = self.r_t[smai]
        self.Lm = self.r_t[smii]
        self.az = -90.0+180.0/64*azi

        # 3.7 Main Subject for Specific Settings
        subj = bs.unpack_from('u2', msg, i)[0]
        i += 2

        if subj == 0:  # Improved Resolution of Main Ellipse (B1)
            # b0-2: Refined Latitude of centre of main ellipse
            # b3-5: Refined Longitude of centre of main ellipse
            # b6-8: Refined Length of semi-major axis
            # b9-11: Refined Length of semi-minor axis
            c1, c2, c3, c4 = bs.unpack_from('u3u3u3u3', msg, i)
            i += 15
            c1 *= 0.000343
            c2 *= 0.000343325

            self.lat += c1
            self.lon += c2

            c3 *= 0.125
            c4 *= 0.125

            d = self.r_t[smai]-self.r_t[smai-1] if smai > 0 else self.r_t[0]
            self.LM -= c3*d

            d = self.r_t[smii]-self.r_t[smii-1] if smii > 0 else self.r_t[0]
            self.Lm -= c4*d

        elif subj == 1:  # Position of the Centre of the Hazard (B2)
            c5, c6 = bs.unpack_from('u7u7', msg, i)
            i += 15

            idx = c5 if c5 <= 63 else c5+1
            self.dlat = -10.0 + idx*20/128
            idx = c6 if c6 <= 63 else c6+1
            self.dlon = -10.0 + idx*20/128

        elif subj == 2:  # Secondary Ellipse Definition (B3)
            c7, c8, c9, c10 = bs.unpack_from('u2u3u5u5', msg, i)
            i += 15

            self.dist = c7*self.LM
            self.LMs = self.LM*(c8+1)*0.25
            self.Lms = self.Lm*(c8+1)*0.25
            self.theta = c9*11.25
            # Guidance Library for Second Ellipse (C10): in Tab for C10

        # Quantitative and detailed information about the Hazard (B4)
        elif subj == 3:
            if hcat == THazard.EARTHQUAKE:  # Earthquake
                d1i, d2i, d3i, d4i = bs.unpack_from('u4u3u4u4', msg, i)
                # D1 – Magnitude on Richter Scale
                self.d1 = self.d1_t[d1i]
                # D2 – Seismic Coefficient
                self.d2 = self.d2_t[d2i]
                # D3 – Azimuth from Centre of Main Ellipse to Epicentre
                self.d3 = d3i*22.5
                # D4 – Vector Length between Centre of
                #      Main Ellipse and Epicentre
                self.d4 = self.d4_t[d4i]

            elif hcat in (THazard.TIDAL_WAVE, THazard.TSUNAMI):
                # Tsunami
                d5i = bs.unpack_from('u3', msg, i)[0]
                # D5 – Wave Height
                self.d5 = self.d5_t[d5i]

            elif hcat in (THazard.COLD_WAVE, THazard.HEAT_WAVE):
                # Cold wave / heat wave
                d6i = bs.unpack_from('u4', msg, i)[0]
                self.d6 = self.d6_t[d6i]

            elif hcat == THazard.HURRICANE:  # Tropical cyclone (hurricane)
                d7i, d8i, d9i = bs.unpack_from('u3u4u3', msg, i)

                self.d7 = self.d7_t[d7i]
                self.d8 = self.d8_t[d8i]
                self.d9 = self.d9_t[d9i]

            elif hcat == THazard.TYPHOON:  # Tropical cyclone (typhoon)
                d36i, d8i, d9i = bs.unpack_from('u3u4u3', msg, i)
                self.d36 = self.d36_t[d36i]
                self.d8 = self.d8_t[d8i]
                self.d9 = self.d9_t[d9i]

            elif hcat == THazard.TORNADO:  # Tornado
                d8i, d9i, d11i = bs.unpack_from('u4u3u3', msg, i)
                self.d8 = self.d8_t[d8i]
                self.d9 = self.d9_t[d9i]
                self.d11 = self.d11_t[d11i]

            elif hcat == THazard.STORM:  # Storm or thunderstorm
                d8i, d9i, d10i, d16i = bs.unpack_from('u4u3u3u3', msg, i)
                self.d8 = self.d8_t[d8i]
                self.d9 = self.d9_t[d9i]
                self.d10 = self.d10_t[d10i]
                self.d16 = self.d16_t[d16i]

            elif hcat == THazard.HAIL:  # Hail
                d12i = bs.unpack_from('u3', msg, i)[0]
                self.d12 = self.d12_t[d12i]

            elif hcat == THazard.RAINFALL:  # Rainfall
                d9i, d13i = bs.unpack_from('u3u4', msg, i)
                self.d9 = self.d9_t[d9i]
                self.d13 = self.d13_t[d13i]
            elif hcat == THazard.SNOWFALL:  # Snowfall
                d14i, d13i = bs.unpack_from('u5u4', msg, i)
                self.d14 = self.d14_t[d14i]
                self.d13 = self.d13_t[d13i]

            elif hcat == THazard.FLOOD:  # Flood
                d15i = bs.unpack_from('u2', msg, i)[0]
                self.d15 = self.d15_t[d15i]

            elif hcat == THazard.LIGHTNING:  # Lightning
                d16i = bs.unpack_from('u3', msg, i)[0]
                self.d16 = self.d16_t[d16i]

            elif hcat == THazard.WINDCHILL:  # Wind chill/frost
                d8i, d6i = bs.unpack_from('u4u4', msg, i)
                self.d8 = self.d8_t[d8i]
                self.d6 = self.d6_t[d6i]

            elif hcat == THazard.DERECHO:  # Derecho
                d8i, d9i, d16i, d11i = bs.unpack_from('u4u3u3u3', msg, i)
                self.d8 = self.d8_t[d8i]
                self.d9 = self.d9_t[d9i]
                self.d16 = self.d16_t[d16i]
                self.d11 = self.d11_t[d11i]

            elif hcat == THazard.FOG:  # Fog
                d17i, d13i = bs.unpack_from('u3u4', msg, i)
                self.d17 = self.d17_t[d17i]
                self.d13 = self.d13_t[d13i]

            elif hcat == THazard.SNOWSTORM:  # Snow storm/blizzard
                d13i, d8i = bs.unpack_from('u4u4', msg, i)
                self.d13 = self.d13_t[d13i]
                self.d8 = self.d8_t[d8i]
            elif hcat == THazard.DROUGHT:  # Drought
                d18i = bs.unpack_from('u2', msg, i)[0]
                self.d18 = self.d18_t[d18i]
            elif hcat == THazard.AVALANCHE_RISK:  # Avalanche risk
                d19i = bs.unpack_from('u3', msg, i)[0]
                self.d19 = self.d19_t[d19i]

            elif hcat == THazard.ASH_FALL:  # Ash fall
                d20i = bs.unpack_from('u3', msg, i)[0]
                self.d20 = self.d20_t[d20i]
            elif hcat == THazard.WINDSURGE:  # Wind/wave/storm surge
                d8i, d5i = bs.unpack_from('u4u3', msg, i)
                self.d8 = self.d8_t[d8i]
                self.d5 = self.d5_t[d5i]
            elif hcat == THazard.GEOMAGNETIC_STORM:  # Geomagnetic/solar storm
                d21i = bs.unpack_from('u3', msg, i)[0]
                self.d21 = self.d21_t[d21i]
            elif hcat == THazard.TERRORISM:  # Terrorism
                d22i = bs.unpack_from('u3', msg, i)[0]
                self.d22 = self.d22_t[d22i]
            elif hcat == THazard.FOREST_FIRE:  # Forest fire / risk of fire
                d23i = bs.unpack_from('u3', msg, i)[0]
                self.d23 = self.d23_t[d23i]
            elif hcat == THazard.CONTAMINATED_WATER:
                # Contaminated drinking water / marine pollution
                d24i = bs.unpack_from('u3', msg, i)[0]
                self.d24 = self.d24_t[d24i]
            elif hcat == THazard.UV_RADIATION:  # UV radiation
                d25i = bs.unpack_from('u4', msg, i)[0]
                self.d25 = self.d25_t[d25i]
            elif hcat == THazard.PANDEMIC:  # Risk of infection / pandemic
                d26i, d35i = bs.unpack_from('u5u6', msg, i)
                self.d26 = self.d26_t[d26i]
                self.d35 = self.d35_t[d35i]
            elif hcat == THazard.NOISE_POLLUTION:  # Noise pollution
                d27i = bs.unpack_from('u4', msg, i)[0]
                self.d27 = self.d27_t[d27i]
            elif hcat == THazard.AIR_POLLUTION:  # Air pollution
                d28i = bs.unpack_from('u3', msg, i)[0]
                self.d28 = self.d28_t[d28i]
            elif hcat == THazard.MARINE_POLLUTION:  # Marine/river pollution
                d24i = bs.unpack_from('u3', msg, i)[0]
                self.d24 = self.d24_t[d24i]
            elif hcat == THazard.OUTAGE_OF_IT:
                # outage of gas supply/IT systems/power systems/
                # emergency number/telephone line
                d29i = bs.unpack_from('u5', msg, i)[0]
                self.d29 = self.d29_t[d29i]
            elif hcat == THazard.NUCLEAR_HAZARD:
                # Radiological hazard, nuclear hazard and nuclear
                # power station accident
                d30i = bs.unpack_from('u4', msg, i)[0]
                self.d30 = self.d30_t[d30i]
            elif hcat == THazard.CHEMICAL_HAZARD:  # Chemical hazard
                d31i = bs.unpack_from('u4', msg, i)[0]
                self.d31 = self.d31_t[d31i]
            elif hcat == THazard.BIO_HAZARD:  # Biological hazard
                d32i, d33i = bs.unpack_from('u2u2', msg, i)
                self.d32 = self.d32_t[d32i]
                self.d33 = self.d33_t[d33i]
            elif hcat == THazard.EXPLOSIVE_HAZARD:  # Explosive hazard
                d34i = bs.unpack_from('u2', msg, i)[0]
                self.d34 = self.d34_t[d34i]

            i += 15

        if self.monlevel > 0:
            print(f"[DCX] {time2str(self.th)} hcat={hcat} " +
                  f"pid={self.pid} {self.severity} {self.hazard} " +
                  f"inst={inst} info={info} {s} ")

            if lati > 0 or loni > 0:
                print(f"lat={self.lat:.4f} lon={self.lon:.4f} " +
                      f"LM={self.LM:.4f} Lm={self.Lm:.4f} az={self.az}")

        return i
