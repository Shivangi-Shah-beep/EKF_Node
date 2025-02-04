#!/usr/bin/env python3

import sympy as sp
import numpy as np

class MeasurementModel:
    def __init__(self, landmark_params, camera_params, transform_params):
        # Store parameters
        self.landmark_params = landmark_params
        self.camera_params = camera_params
        self.transform_params = transform_params

        # Derive the measurement model
        self._derive_measurement_model()

    def _derive_measurement_model(self):
        # Symbolic variables
        x, y, theta = sp.symbols('x y theta', real=True)
        x_l, y_l, r_l, h_l = sp.symbols('x_l y_l r_l h_l', real=True)
        t_cx, t_cy, t_cz = sp.symbols('t_cx t_cy t_cz', real=True)
        f_x, f_y, c_x, c_y = sp.symbols('f_x f_y c_x c_y', real=True)

        # Camera position in global frame
        x_c = x + sp.cos(theta) * t_cx - sp.sin(theta) * t_cy
        y_c = y + sp.sin(theta) * t_cx + sp.cos(theta) * t_cy

        # Bearing angle psi
        delta_x = x_l - x_c
        delta_y = y_l - y_c
        psi = sp.atan2(delta_y, delta_x)

        # Feature points in global frame
        x1 = x_l - r_l * sp.sin(psi)
        y1 = y_l + r_l * sp.cos(psi)
        x2 = x_l + r_l * sp.sin(psi)
        y2 = y_l - r_l * sp.cos(psi)
        z0 = 0
        z1 = h_l

        p1_g = sp.Matrix([x1, y1, z0, 1])
        p2_g = sp.Matrix([x2, y1, z0, 1])
        p3_g = sp.Matrix([x2, y2, z1, 1])
        p4_g = sp.Matrix([x1, y2, z1, 1])

        # Transformation matrices
        T_mr = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta), 0, x],
            [sp.sin(theta), sp.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T_ro = sp.Matrix([
            [0, 0, 1, t_cx],
            [-1, 0, 0, t_cy],
            [0, -1, 0, t_cz],
            [0, 0, 0, 1]
        ])

        T_mo = T_mr * T_ro
        T_om = sp.simplify(T_mo.inv())

        # Transform feature points to optical frame
        p1_o = T_om * p1_g
        p2_o = T_om * p2_g
        p3_o = T_om * p3_g
        p4_o = T_om * p4_g

        # Camera projection matrix
        P = sp.Matrix([
            [f_x, 0, c_x, 0],
            [0, f_y, c_y, 0],
            [0, 0, 1, 0]
        ])

        # Project points to pixel space
        def project_point(p_o):
            u_v_w = P * p_o
            u = u_v_w[0] / u_v_w[2]
            v = u_v_w[1] / u_v_w[2]
            return sp.Matrix([u, v])

        # Project each point
        p1_p = project_point(p1_o)
        p2_p = project_point(p2_o)
        p3_p = project_point(p3_o)
        p4_p = project_point(p4_o)

        # Measurement vector (z_pred)
        z_pred_sym = sp.Matrix([
            p1_p[0], p1_p[1],
            p2_p[0], p2_p[1],
            p3_p[0], p3_p[1],
            p4_p[0], p4_p[1]
        ])

        # Jacobian with respect to state variables x, y, theta
        H_sym = z_pred_sym.jacobian([x, y, theta])

        # Lambdify the expressions for numerical computation
        params = (x, y, theta, t_cx, t_cy, t_cz,
                  f_x, f_y, c_x, c_y,
                  x_l, y_l, r_l, h_l)
        self.z_pred_func = sp.lambdify(params, z_pred_sym, 'numpy')
        self.H_func = sp.lambdify(params, H_sym, 'numpy')

    def jacobian(self, state_vector):
        x, y, theta = state_vector

        # Unpack parameters
        t_cx = self.transform_params['t_cx']
        t_cy = self.transform_params['t_cy']
        t_cz = self.transform_params['t_cz']
        f_x = self.camera_params['fx']
        f_y = self.camera_params['fy']
        c_x = self.camera_params['cx']
        c_y = self.camera_params['cy']
        x_l = self.landmark_params['x_l']
        y_l = self.landmark_params['y_l']
        r_l = self.landmark_params['r_l']
        h_l = self.landmark_params['h_l']

        # Compute the Jacobian numerically
        H_numeric = self.H_func(
            x, y, theta,
            t_cx, t_cy, t_cz,
            f_x, f_y, c_x, c_y,
            x_l, y_l, r_l, h_l
        )
        return H_numeric

    def measurement(self, state_vector, observed_features, variances):
        x, y, theta = state_vector

        # Unpack parameters
        t_cx = self.transform_params['t_cx']
        t_cy = self.transform_params['t_cy']
        t_cz = self.transform_params['t_cz']
        f_x = self.camera_params['fx']
        f_y = self.camera_params['fy']
        c_x = self.camera_params['cx']
        c_y = self.camera_params['cy']
        x_l = self.landmark_params['x_l']
        y_l = self.landmark_params['y_l']
        r_l = self.landmark_params['r_l']
        h_l = self.landmark_params['h_l']

        # Compute expected measurements
        z_pred = self.z_pred_func(
            x, y, theta,
            t_cx, t_cy, t_cz,
            f_x, f_y, c_x, c_y,
            x_l, y_l, r_l, h_l
        )

        # Ensure z_pred is a numpy array
        z_pred = np.array(z_pred).flatten()

        # Prepare actual measurements
        z_actual = np.array(observed_features)

        # Handle missing measurements
        if len(z_actual) < 8:
        # Substitute missing measurements with corresponding z_pred values
            z_actual = np.concatenate([z_actual, z_pred[len(z_actual):]])

    # Handle missing variances
        if not variances:
            variances = [10.0] * 8  # Default variances if none are provided
        elif len(variances) < 8:
            # If variances are fewer than 8, pad with the last value
            variances = np.concatenate([variances, [variances[-1]] * (8 - len(variances))])

        # Measurement covariance matrix
        R = np.diag(variances)
        # Compute difference
        z_diff = z_actual - z_pred

        return z_pred, z_actual, R, z_diff
