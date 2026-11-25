import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution, minimize
import pandas as pd
import symmetry as sym
import simulation as simul

import warnings
warnings.filterwarnings("ignore")

class refinement():

    def __init__(self, lattice_constants, lattice_angles, atoms, magnetic_atoms, posits, occs, Us, magnetic_moments, scatt_lengths, magnetic_form_coefs):

        self.lattice_constants = lattice_constants
        self.alpha = lattice_angles[0]
        self.beta = lattice_angles[1]
        self.gamma = lattice_angles[2]

        self.atoms = atoms
        self.magnetic_atoms = magnetic_atoms
        self.posits = posits
        self.occs = occs
        self.Us = Us
        self.magnetic_moments = magnetic_moments
        self.scatt_lengths = scatt_lengths
        self.magnetic_form_coefs = magnetic_form_coefs

        self.unit_vectors = np.zeros((3, 3), )
        self.unit_vectors[0] = np.array([1.0, 0.0, 0.0])
        self.unit_vectors[1] = np.array([np.cos(np.pi*self.gamma/180), np.sin(np.pi*self.gamma/180), 0.0])    
        self.unit_vectors[2][0] = np.cos(np.pi*self.beta/180)
        self.unit_vectors[2][1] = (np.cos(np.pi*self.alpha/180) - np.cos(np.pi*self.beta/180)*np.cos(np.pi*self.gamma/180))/np.sin(np.pi*self.gamma/180)
        self.unit_vectors[2][2] = np.sqrt(max(0.0, 1 - np.cos(np.pi*self.beta/180)**2 - ((np.cos(np.pi*self.alpha/180) - np.cos(np.pi*self.beta/180)*np.cos(np.pi*self.gamma/180))/np.sin(np.pi*self.gamma/180))**2))
        
        self.V_cell = np.prod(lattice_constants)*np.dot(np.cross(self.unit_vectors[0], self.unit_vectors[1]), self.unit_vectors[2])

    def import_data(self, directory):

        data = pd.read_csv(directory, sep=r"\s+", 
                             usecols = [0, 1, 2, 3, 4, 5, 6, 7, 18], 
                             names = ["h", "k", "l", "Fsqr", "err", "group", "wavelength", "tbar", "ttheta"])

        data = data[~((data["h"] == 0) &
              (data["k"] == 0) &
              (data["l"] == 0))]

        print("Peaks number:", len(data))

        x = data.iloc[:, 0:3].reset_index(drop=True)
        y = data.iloc[:, 3].reset_index(drop=True)
        y_err = data.iloc[:, 4].reset_index(drop=True)
        groups = data.iloc[:, 5].reset_index(drop=True) - 1
        self.group_index = max(groups)
        wavelength = data.iloc[:, 6].reset_index(drop=True)
        tbar = data.iloc[:, 7].reset_index(drop=True)
        ttheta = data.iloc[:, 8].reset_index(drop=True)
        
        sorted_idx = np.argsort(y.to_numpy()) 
        
        self.x = x.iloc[sorted_idx].reset_index(drop=True)
        self.y = y.iloc[sorted_idx].reset_index(drop=True)
        self.y_err = y_err.iloc[sorted_idx].reset_index(drop=True)
        self.groups = groups.iloc[sorted_idx].reset_index(drop=True)
        self.wavelength = wavelength.iloc[sorted_idx].reset_index(drop=True)
        self.tbar = tbar.iloc[sorted_idx].reset_index(drop=True)
        self.ttheta = ttheta.iloc[sorted_idx].reset_index(drop=True)

    def init_refinement(self, MSG, constraints = [], print_freedoms = True):

        self.MSG = MSG
        self.constraints = constraints

        self.magnetic_moments_fitting = copy.deepcopy(self.magnetic_moments)
        self.occs_fitting = copy.deepcopy(self.occs)
        self.posits_fitting = copy.deepcopy(self.posits)
        self.Us_fitting = copy.deepcopy(self.Us) 
        self.A_fitting = 0.5*np.ones(self.group_index+1,)
        self.g_fitting = 0.0
        self.a_fitting = 0.0
        self.b_fitting = 0.0

        self.n_wp = 2

        self.nonmag_minimal_operators = {}
        self.mag_minimal_operators = {}
        self.U_matricies = {}

        for atom in self.atoms:
        
            self.nonmag_minimal_operators[atom] = sym.get_minimal_operators(self.posits_fitting[atom], self.MSG, self.lattice_constants, magnetic = False)
        
        for atom in self.atoms:
        
            self.U_matricies[atom] = sym.get_minimal_matrices_U(self.posits_fitting[atom], self.MSG, self.lattice_constants)
            
        for atom in self.magnetic_atoms:
        
            self.mag_minimal_operators[atom] = sym.get_minimal_operators(self.posits_fitting[atom], self.MSG, self.lattice_constants, magnetic = True)

        self.A_scale = np.mean(self.y)/np.mean(simul.intensity_hkl(self.x, self.lattice_constants, self.posits_fitting, self.occs_fitting, self.Us_fitting, 
                                                             self.magnetic_moments_fitting, np.ones(self.group_index+1,),
                                      self.groups, self.nonmag_minimal_operators, self.U_matricies, self.mag_minimal_operators, self.unit_vectors, 
                                                                  self.scatt_lengths, self.magnetic_form_coefs))*2
        
        self.b_scale = np.median(self.y)
        self.g_scale = 1e4
        self.m_scale = 100
        self.occ_scale = 10

        self.free_positions = {}
        self.free_Us = {}
        self.free_moments = {}
        
        for atom in self.atoms:
        
            self.free_positions[atom] = sym.solve_free_positions(self.posits_fitting[atom], self.MSG)
            self.free_Us[atom] = sym.solve_free_U(self.posits_fitting[atom], self.MSG, self.lattice_constants)
        
        for atom in self.magnetic_atoms:
        
            self.free_moments[atom] = sym.solve_free_moments(self.posits_fitting[atom], self.MSG, self.lattice_constants)

        self.set_params = {}
        self.params = {}
        self.params_err = {}
        self.bounds = {}

        self.keys = ["A"]

        for atom in self.magnetic_atoms:

            for i in range(len(self.free_moments[atom])):

                key = f"{atom}_m_{i+1}"
                self.keys.append(key)

        for atom in self.atoms:

            key = f"{atom}_occ"
            self.keys.append(key)

            for i in range(len(self.free_positions[atom])):

                key = f"{atom}_posit_{i+1}"
                self.keys.append(key)

            for i in range(len(self.free_Us[atom])):

                key = f"{atom}_U_{i+1}"
                self.keys.append(key)

        self.keys.extend(["g", "a", "b"])

        self.fitting_to_params()

        for key in self.keys:

            self.set_params[key] = True

        self.params_err["A"] = np.zeros(self.group_index+1,)
        self.params_err["a"] = np.nan
        self.params_err["b"] = np.nan

        for key in self.keys:

            if key not in ["A", "a", "b"]:

                 self.params_err[key] = 0.0

        self.bounds["A"] = [0.2, 1.5]
        self.bounds["g"] = [0, 0.1]
        self.bounds["a"] = [0, 0.2]
        self.bounds["b"] = [0, 0.1]

        for key in self.keys:

            if key not in ["A", "a", "b", "g"]:

                key_split = key.split("_")

                if key_split[1] == "m":

                    self.bounds[key] = [-10, 10]

                if key_split[1] == "occ":

                    self.bounds[key] = [0.0, 1.0]

                if key_split[1] == "U":

                    self.bounds[key] = [-0.1, 0.1]

                if key_split[1] == "posit":

                    self.bounds[key] = [-0.05, 0.05]

        self.apply_constraints()
        self.params_to_fitting()
        self.update_active_parameters()

        if print_freedoms == True:

            print("Active keys:")
            print(self.active_keys)
            print("\n")
            print("Free displacements:")
            for atom in self.atoms:
                for i in range(len(self.free_positions[atom])):
                    print(f"{atom}_posit_{i+1}: {self.free_positions[atom][i]}")
            print("\n")
            print("Free U elements:")
            for atom in self.atoms:
                for i in range(len(self.free_Us[atom])):
                    print(f"{atom}_U_{i+1}: {self.free_Us[atom][i]}")
            print("\n")
            print("Free magnetic moments:")
            for atom in self.magnetic_atoms:
                for i in range(len(self.free_moments[atom])):
                    print(f"{atom}_m_{i+1}: {self.free_moments[atom][i]}")
            
                
    def fitting_to_params(self, ):

        m_projections = {}
        posit_projections = {}
        U_projections = {}

        for atom in self.magnetic_atoms:

            if len(self.free_moments[atom]) != 0:

                E = self.unit_vectors.T
                target = E @ np.asarray(self.magnetic_moments_fitting[atom])
                basis = E @ np.asarray(self.free_moments[atom]).T
                m_projections[atom] = np.linalg.lstsq(basis, target, rcond=None)[0]

        for atom in self.atoms:

            if len(self.free_positions[atom]) != 0:

                target = np.asarray(self.posits_fitting[atom]) - np.asarray(self.posits[atom])
                basis = np.asarray(self.free_positions[atom]).T
                posit_projections[atom] = np.linalg.lstsq(basis, target, rcond=None)[0]

            if len(self.free_Us[atom]) != 0:

                target = np.asarray(self.Us_fitting[atom])
                basis = np.asarray(self.free_Us[atom]).T
                U_projections[atom] = np.linalg.lstsq(basis, target, rcond=None)[0]

        for key in self.keys:
        
            if key not in ["A", "a", "b", "g"]:

                key_split = key.split("_")
                atom = key_split[0]

                if key_split[1] == "m":

                    index = int(key_split[2]) - 1                   
                    self.params[key] = m_projections[atom][index]

                if key_split[1] == "posit":

                    index = int(key_split[2]) - 1                   
                    self.params[key] = posit_projections[atom][index]

                if key_split[1] == "U":

                    index = int(key_split[2]) - 1                   
                    self.params[key] = U_projections[atom][index]

                if key_split[1] == "occ":
               
                    self.params[key] = self.occs_fitting[atom]  

        self.params["A"] = self.A_fitting
        self.params["g"] = self.g_fitting
        self.params["a"] = self.a_fitting
        self.params["b"] = self.b_fitting

    def update_active_parameters(self,):

        self.active_scale_params = list(self.params["A"])
        self.active_scale_bounds = [self.bounds["A"] for _ in range(self.group_index + 1)]
        
        self.active_keys = [k for k, v in self.set_params.items() if k != "A" and v]
        
        self.active_nonscale_params = []
        self.active_nonscale_bounds = []

        for k in self.active_keys:

            if k not in ["a", "b", "g"]:

                par = k.split("_")[1]

                if par == "m":
    
                    bound = list(np.asarray(self.bounds[k])/self.m_scale)
                    param = self.params[k]/self.m_scale
                    self.active_nonscale_bounds.append(bound)
                    self.active_nonscale_params.append(param)
    
                elif par == "occ":
    
                    bound = list(np.asarray(self.bounds[k])/self.occ_scale)
                    param = self.params[k]/self.occ_scale
                    self.active_nonscale_bounds.append(bound)
                    self.active_nonscale_params.append(param)

                else:

                    self.active_nonscale_bounds.append(self.bounds[k])
                    self.active_nonscale_params.append(self.params[k])

            else:

                self.active_nonscale_bounds.append(self.bounds[k])
                self.active_nonscale_params.append(self.params[k])

        self.active_params = self.active_scale_params + self.active_nonscale_params
        self.active_bounds = self.active_scale_bounds + self.active_nonscale_bounds

        self.n_wp = 0

        for key in ["a", "b"]:

            if self.set_params[key] == True:

                self.n_wp +=1


    def params_to_fitting(self, ):

        self.A_fitting = self.params["A"]
        self.a_fitting = self.params["a"]
        self.b_fitting = self.params["b"] 
        self.g_fitting = self.params["g"]

        for atom in self.magnetic_atoms:

            number = len(self.free_moments[atom])
            moment = np.zeros((3,))

            for i in range(number):

                moment += self.params[f"{atom}_m_{i+1}"]*np.asarray(self.free_moments[atom][i])

            self.magnetic_moments_fitting[atom] = list(moment)

        for atom in self.atoms:

            number = len(self.free_positions[atom])
            posit = np.zeros((3,))

            for i in range(number):

                posit += self.params[f"{atom}_posit_{i+1}"]*np.asarray(self.free_positions[atom][i])

            self.posits_fitting[atom] = list(np.asarray(self.posits[atom]) + posit)

        for atom in self.atoms:

            number = len(self.free_Us[atom])
            U = np.zeros((6,))

            for i in range(number):

                U += self.params[f"{atom}_U_{i+1}"]*np.asarray(self.free_Us[atom][i])

            self.Us_fitting[atom] = list(U)

        for atom in self.atoms:
            
            self.occs_fitting[atom] = self.params[f"{atom}_occ"]

    def apply_constraints(self, ):

        for constraint in self.constraints:

            constraint_split = constraint.split("_")

            if constraint_split[0] == "UEqual":

                atom1 = constraint_split[1]
                atom2 = constraint_split[2]

                for i in range(len(self.free_Us[atom2])):

                    self.set_params[f"{atom2}_U_{i+1}"] = False
                    self.params[f"{atom2}_U_{i+1}"] = copy.deepcopy(self.params[f"{atom1}_U_{i+1}"])

                self.Us_fitting[atom2] = copy.deepcopy(self.Us_fitting[atom1])

            if constraint_split[0] == "positEqual":

                atom1 = constraint_split[1]
                atom2 = constraint_split[2]

                for i in range(len(self.free_positions[atom2])):

                    self.set_params[f"{atom2}_posit_{i+1}"] = False
                    self.params[f"{atom2}_posit_{i+1}"] = copy.deepcopy(self.params[f"{atom1}_posit_{i+1}"])

                self.posits_fitting[atom2] = copy.deepcopy(self.posits_fitting[atom1])

            if constraint_split[0] == "occSum":

                atom1 = constraint_split[1]
                atom2 = constraint_split[2]

                summation = float(constraint_split[3])
                self.set_params[f"{atom2}_occ"] = False
                self.params[f"{atom2}_occ"] = summation - self.params[f"{atom1}_occ"]
                self.occs_fitting[atom2] = summation - self.occs_fitting[atom1]

            if constraint_split[0] == "exclude":

                atom = constraint_split[1]
                for key in self.keys:

                    if key not in ["A", "g", "a", "b"]:

                        key_split = key.split("_")
    
                        if key_split[0] == atom:
    
                            self.set_params[key] = False
                            self.params[key] = 0.0

            if constraint_split[0] == "positFix":
            
                target_atoms = self.atoms if constraint_split[1] == "all" else [constraint_split[1]]
            
                for atom in target_atoms:
            
                    for i in range(len(self.free_positions[atom])):
            
                        key = f"{atom}_posit_{i+1}"
            
                        self.set_params[key] = False
                        self.params[key] = 0.0

    def magnetic_simul(self, params):

        self.params["A"] =  params[:self.group_index+1]

        physical_keys = [k for k in self.active_keys if k not in ["a", "b"]]
        
        for i, par in enumerate(physical_keys):
            
            idx = i+self.group_index+1

            if par != "g":

                if par.split("_")[1] == "m":
                    self.params[par] = params[idx]*self.m_scale
                elif par.split("_")[1] == "occ":
                    self.params[par] = params[idx]*self.occ_scale
                else:
                    self.params[par] = params[idx]

            else:

                self.params[par] = params[idx]

        self.apply_constraints()
        self.params_to_fitting() 

        intensity = simul.intensity_hkl_extinction(self.x, self.lattice_constants, self.posits_fitting, self.occs_fitting, self.Us_fitting, self.magnetic_moments_fitting, 
                                                   self.A_scale*self.A_fitting, self.groups, self.wavelength, self.ttheta, self.tbar, self.g_scale*self.g_fitting, 
                                                   self.nonmag_minimal_operators, self.U_matricies, self.mag_minimal_operators, self.unit_vectors, self.scatt_lengths, 
                                                   self.magnetic_form_coefs, self.V_cell)         
        
        return intensity

    def weight(self, intensity, a, b):

        return 1/(self.y_err**2 + (a*intensity)**2 + self.b_scale*b*intensity)

    def magnetic_fit_residual(self, params):
    
        intensity = self.magnetic_simul(params[:len(params) - self.n_wp])

        i = 0

        for key in ["a", "b"]:

            if self.set_params[key] == True:

                i += 1
                self.params[key] = params[len(params) - self.n_wp + i - 1]

        self.a_fitting = self.params["a"]
        self.b_fitting = self.params["b"]

        w = self.weight(intensity, self.a_fitting, self.b_fitting)
        
        return np.sqrt(w)*(self.y - intensity)
    
    def magnetic_fit_chi2(self, params):
        
        wresidual = self.magnetic_fit_residual(params)
        return np.sum(wresidual**2) / (len(self.y) - len(params))

    def likelihood_objective(self, params):
    
        intensity = self.magnetic_simul(params[:len(params) - self.n_wp])
        i = 0

        for key in ["a", "b"]:

            if self.set_params[key] == True:

                i += 1
                self.params[key] = params[len(params) - self.n_wp + i - 1]

        self.a_fitting = self.params["a"]
        self.b_fitting = self.params["b"]
        
        w = self.weight(intensity, self.a_fitting, self.b_fitting)
    
        return np.mean(w*(self.y - intensity)**2 - np.log(w))

    def least_square_residual(self, params):
    
        intensity = self.magnetic_simul(params)
        
        return np.sqrt(self.w_fixed)*(self.y - intensity)

    def conduct_differential_evolution(self,):

        self.update_active_parameters()
        self.results = differential_evolution(self.likelihood_objective, bounds=self.active_bounds, atol = 1e-2, tol = 0.0, polish=False)
        
        self.chi2 = self.magnetic_fit_chi2(self.results.x)
        indices = self.y > 0
        self.R = np.abs(np.sqrt(self.y[indices])-np.sqrt(self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp])[indices])).sum()*100/np.sqrt(self.y[indices]).sum()

        i = 0

        for key in ["a", "b"]:

            if self.set_params[key] == True:

                i += 1
                self.params[key] = self.results.x[len(self.results.x) - self.n_wp + i - 1]

        self.a_fitting = self.params["a"]
        self.b_fitting = self.params["b"]

        w = self.weight(self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp]), self.a_fitting, self.b_fitting)
        self.wR2 = 100*np.sqrt(np.sum(w*(self.y-self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp]))**2)/np.sum(w*self.y**2))
        self.NLL = self.likelihood_objective(self.results.x)

    def conduct_minimize_ls(self,):

        self.update_active_parameters()

        self.results = minimize(self.likelihood_objective, x0=self.active_params, bounds=self.active_bounds, method="L-BFGS-B")
        self.chi2 = self.magnetic_fit_chi2(self.results.x)
        indices = self.y > 0
        self.R = np.abs(np.sqrt(self.y[indices])-np.sqrt(self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp])[indices])).sum()*100/np.sqrt(self.y[indices]).sum()

        i = 0

        for key in ["a", "b"]:

            if self.set_params[key] == True:

                i += 1
                self.params[key] = self.results.x[len(self.results.x) - self.n_wp + i - 1]

        self.a_fitting = self.params["a"]
        self.b_fitting = self.params["b"]

        w = self.weight(self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp]), self.a_fitting, self.b_fitting)
        self.wR2 = 100*np.sqrt(np.sum(w*(self.y-self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp]))**2)/np.sum(w*self.y**2))
        self.NLL = self.likelihood_objective(self.results.x)

        intensity = self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp])
        self.w_fixed = self.weight(intensity, self.params["a"], self.params["b"])

        weight_flags = {key: self.set_params[key] for key in ["a", "b"]}
        param_flags = copy.deepcopy(self.params)

        self.set_params["a"] = False
        self.set_params["b"] = False
        
        self.update_active_parameters()

        lb = [b[0] for b in self.active_bounds]
        ub = [b[1] for b in self.active_bounds]

        self.results_ls = least_squares(self.least_square_residual, x0=self.active_params, bounds=(lb, ub), jac="3-point", max_nfev=1)
        wresidual = self.least_square_residual(self.results_ls.x)
        dof = len(self.y) - len(self.results_ls.x)

        J = self.results_ls.jac
        cov = np.linalg.pinv(J.T @ J)
        self.stderr = np.sqrt(np.diag(cov))

        self.params_err["A"] = np.array(self.stderr[:self.group_index+1])

        for i, k in enumerate(self.active_keys):

            if k == "g":

                self.params_err[k] = self.stderr[self.group_index+1+i]

            else:

                scale = 1

                if k.split("_")[1] == "m":

                    scale = self.m_scale

                if k.split("_")[1] == "occ":

                    scale = self.occ_scale

                self.params_err[k] = scale*self.stderr[self.group_index+1+i]

        for key, active in weight_flags.items():
            self.set_params[key] = active

        for key, value in param_flags.items():
            self.params[key] = value

        self.update_active_parameters()
        self.params_to_fitting()
        
    def process_results(self, print_out = False, return_out = False, save = False, filename = ""):

        def format_with_error(value, error, digits=2):
            
            if not np.isfinite(error) or abs(error) < 1e-5:
                return f"{value:.5f}"
        
            error = abs(error)
        
            decimals = digits - 1 - int(np.floor(np.log10(error)))
            decimals = max(0, decimals)
        
            error_int = int(round(error * 10**decimals))
        
            return f"{value:.{decimals}f}({error_int})"

        def U_eigenvalues(U):
        
            U11, U22, U33, U23, U13, U12 = U
        
            U_matrix = np.array([
                [U11, U12, U13],
                [U12, U22, U23],
                [U13, U23, U33]
            ])
        
            E = np.asarray(self.unit_vectors).T
        
            U_cart = E @ U_matrix @ E.T
        
            return np.linalg.eigvalsh(U_cart)

        symmetry_lattice_information = pd.DataFrame([{"MSG": self.MSG, "a": self.lattice_constants[0], "b": self.lattice_constants[1], "c": self.lattice_constants[2],
                                            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}])
        fitting_information = pd.DataFrame([{"gof": f"{np.sqrt(self.chi2):.3f}", "R (%)": f"{self.R:.3f}", "wR2 (%)": f"{self.wR2:.3f}", "tmNLL": f"{self.NLL:.3f}", "a":f"{self.a_fitting:.5f}" , "b":f"{self.b_fitting:.5f}"}])
        structural_information = pd.DataFrame(index = self.atoms, columns = ["x", "y", "z", "U11", "U22", "U33", "U23", "U13", "U12", "U_eigen", "occ"])
        magnetic_information = pd.DataFrame(index = self.magnetic_atoms, columns = ["ma", "mb", "mc"])
        scale_ect_information = pd.DataFrame(index = [f"scale{i+1}" for i in range(self.group_index+1)] + ["g"])

        for atom in self.atoms:

            posit_error = np.zeros((3,))
            U_error = np.zeros((6,))

            for i in range(len(self.free_positions[atom])):
            
                posit_error += (self.params_err[f"{atom}_posit_{i+1}"]*np.asarray(self.free_positions[atom][i]))**2
            
            posit_error = np.sqrt(posit_error)

            for i in range(len(self.free_Us[atom])):
            
                U_error += (self.params_err[f"{atom}_U_{i+1}"]*np.asarray(self.free_Us[atom][i]))**2

            U_error = np.sqrt(U_error)
            
            occ_err = self.params_err[f"{atom}_occ"]

            for i, comp in enumerate(["x", "y", "z"]):
            
                structural_information.loc[atom, comp] = format_with_error(self.posits_fitting[atom][i], posit_error[i])

            for i, comp in enumerate(["U11", "U22", "U33", "U23", "U13", "U12"]):
            
                structural_information.loc[atom, comp] = format_with_error(self.Us_fitting[atom][i], U_error[i])

            U_eigens = U_eigenvalues(self.Us_fitting[atom])

            structural_information.loc[atom, "U_eigen"] = f"{U_eigens[0]:.5f}, {U_eigens[1]:.5f}, {U_eigens[2]:.5f}"

            structural_information.loc[atom, "occ"] = format_with_error(self.occs_fitting[atom], occ_err)

        for atom in self.magnetic_atoms:

            magnetic_error = np.zeros((3, ))

            for i in range(len(self.free_moments[atom])):
            
                magnetic_error += (self.params_err[f"{atom}_m_{i+1}"]*np.asarray(self.free_moments[atom][i]))**2
            
            magnetic_error = np.sqrt(magnetic_error)

            for i, comp in enumerate(["ma", "mb", "mc"]):

                magnetic_information.loc[atom, comp] = format_with_error(self.magnetic_moments_fitting[atom][i], magnetic_error[i])

        for i in range(self.group_index+1):

            scale_ect_information.loc[f"scale{i+1}", "value"] = format_with_error(self.A_fitting[i], self.params_err["A"][i])

        scale_ect_information.loc["g", "value"] = format_with_error(self.g_fitting, self.params_err["g"])

        fitting_params = pd.DataFrame()

        for i in range(len(self.params["A"])):

            fitting_params.loc[f"scale{i+1}", "value"] = self.params["A"][i]
            fitting_params.loc[f"scale{i+1}", "error"] = self.params_err["A"][i]
            fitting_params.loc[f"scale{i+1}", "active"] = self.set_params["A"]

        for key in self.params:

            if key == "A":
                continue

            fitting_params.loc[key, "value"] = self.params[key]
            fitting_params.loc[key, "error"] = self.params_err[key]
            fitting_params.loc[key, "active"] = self.set_params[key]

        if print_out == True:

            print(symmetry_lattice_information)
            print(fitting_information)
            print(magnetic_information)
            print(structural_information)
            print(scale_ect_information)
            print(fitting_params)

        if save == True:

            with pd.ExcelWriter(f"{filename}.xlsx", engine="openpyxl") as writer:

                symmetry_lattice_information.to_excel(writer, sheet_name="symmetry_lattice_information", index=False)
                fitting_information.to_excel(writer, sheet_name="Fitting", index=False)
                magnetic_information.to_excel(writer, sheet_name="Magnetic")
                structural_information.to_excel(writer, sheet_name="Structure")
                scale_ect_information.to_excel(writer, sheet_name="Scale_Extinction")
                fitting_params.to_excel(writer, sheet_name="Fitting_params")

        if return_out == True:

            return symmetry_lattice_information, fitting_information, magnetic_information, structural_information, scale_ect_information, fitting_params
        
    def plot_results(self, save = False, file_name = ""):

        fig, ax = plt.subplots(
            2, 1,
            figsize=(15, 8),
            gridspec_kw={'hspace': 0}
        )
        
        xpos = np.arange(len(self.x))
        
        ax[0].bar(xpos, self.y, color="red", alpha=0.5, width=1.0)
        ax[0].set_xlim(-0.5, len(self.x) - 0.5)
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Intensity (a. u.)")
        ax[0].bar(xpos, self.magnetic_simul(self.results.x[:len(self.results.x) - self.n_wp]), color="blue", alpha=0.5, width=1.0)

        ax[1].set_ylabel("Normalized residual (a. u.)")
        residual = self.magnetic_fit_residual(self.results.x)
        chi2 = residual**2
        ax[1].bar(xpos, residual, color="blue")
        ax[1].set_xlim(-0.5, len(self.x) - 0.5)
        
        # Remove x ticks and labels from both panels
        for a in ax:
            a.set_xticks([])

        if save == True:
            plt.savefig(f"{file_name}_fitting.png", dpi = 300,  bbox_inches = "tight")

        block_len = int(len(chi2)/10)
        boundaries = []
        centers = []
        chi2_bins = []
        
        for i in range(9):
            boundaries.append(block_len*(i+1))
            
        for i in range(10):
            if i == 9:
                upper = len(chi2)
            else:
                upper = boundaries[i]

            if i == 0:
                lower = 0
            else:
                lower = boundaries[i-1]

            center = (lower + upper - 1)/2
            chi2_bin = np.mean(chi2[lower:upper])

            centers.append(center)
            chi2_bins.append(chi2_bin)

        fig, ax = plt.subplots(1,1, figsize = (15, 5))
        ax.plot(centers, chi2_bins, color = "red", linestyle = "-", marker = "o")
        ax.plot(np.arange(len(chi2)), self.chi2*np.ones(len(chi2)), color = "blue", linestyle = "--")
        ax.set_ylim(0, 1.1*max(chi2_bins))
        ax.set_ylabel("Chi2 contribution (a. u.)")
        ax.set_xticks([])

        if save == True:
            plt.savefig(f"{file_name}_Chi2.png", dpi = 300,  bbox_inches = "tight")