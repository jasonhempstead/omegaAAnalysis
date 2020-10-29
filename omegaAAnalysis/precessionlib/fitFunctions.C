// functions needed for fitting the precession histograms
//
// Aaron Fienberg
// September 2018
// triggering rebuild

#include <iostream>

#include "TH1.h"
#include "TF1.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"

// the histogram used for including the muon loss term
TH1D* cumuLossHist = nullptr;

bool lossHistIsInitialized() { return cumuLossHist != nullptr; }

void initializeLossHist(const char* histName) {
  cumuLossHist = (TH1D*)gROOT->FindObject(histName);
}

// do not call this before initializing the loss hist ptr
double cumuLoss(double x) { return cumuLossHist->Interpolate(x); }

// hack to get systematic scans done for February 2019 meeting
int cboEnvNum = 0;
void setCBOEnvelopeNum(int newEnvNum) { cboEnvNum = newEnvNum; }

double cboEnvelope(double t, double param) {
  // test
  const double tau_cbo = param;

  if (cboEnvNum == 0) {
    return exp(-t / tau_cbo);
  }

  else if (cboEnvNum == 1) {
    return exp(-t / tau_cbo) + 0.15;
  }

  else if (cboEnvNum == 2) {
    return exp(-t / tau_cbo) * (1 + 0.135 * cos(0.00874 * t - 0.48));
  }

  else if (cboEnvNum == 3) {
    // gaus because why not
    return exp(-t * t / tau_cbo / tau_cbo);
  }

  else {
    return -1;
  }
}

// CBO frequency models
// the original model used for the 60-Hour Unblinding
// something like this:
// https://muon.npl.washington.edu/elog/g2/Tracking+Analysis/122
// w_cbo is the base cbo frequency, t is the time,
// and p is the param vector
double cbo_model_original(double w_cbo, double t, const double* p) {
  const double dw = p[0] / 100 / 1000;
  const double cbo_exp_a = p[1] / 100;
  const double cbo_exp_b = p[2] / 100;
  const double tau_a = p[3];
  const double tau_b = p[4];

  return w_cbo * (1 + dw * t + cbo_exp_a * exp(-t / tau_a) +
                  cbo_exp_b * exp(-t / tau_b));
}

// second model, provided before the elba meeting
// so far I have only received this via email
double cbo_model_elba(double w_cbo, double t, const double* p) {
  // p[0] unused in this model
  const double cbo_exp_a = p[1];
  const double cbo_exp_b = p[2];
  const double tau_a = p[3];
  const double tau_b = p[4];

  // do not evaluate with small t,
  // where 1 / t explodes
  static constexpr double cutoff_t = 1;
  if (t >= cutoff_t) {
    return w_cbo + cbo_exp_a * exp(-t / tau_a) / t +
           cbo_exp_b * exp(-t / tau_b) / t;
  } else {
    return cbo_model_elba(w_cbo, cutoff_t, p);
  }
}

// approximate is ok ;)
constexpr double omega_c = 42.1153248017936;

// get field-index from CBO frequency
double n_of_omega_CBO(double omega_cbo) {
  return (omega_cbo * (2 * omega_c - omega_cbo)) / pow(omega_c, 2);
}

// constexpr unsigned int n_full_fit_parameters = 27;
constexpr unsigned int n_full_fit_parameters = 33;
double full_wiggle_fit(const double* x, const double* p) {
  const double t = x[0];
  const double N_0 = p[0];
  const double tau = p[1];
  // A, phi will be modified by CBO
  double A = p[2];
  double phi = p[3];
  const double R = p[4];
  const double wa_ref = p[5];
  const double tau_cbo = p[6];
  const double A_cbo = p[7];
  const double phi_cbo = p[8];
  // w_cbo will be modified by tracker model
  double w_cbo = p[9];
  const double tau_vw = p[10];
  const double A_vw = p[11];
  const double phi_vw = p[12];
  // could be updated if we are fitting in
  // "use field index" mode
  double w_vw = p[13];
  const double K_loss = p[14];

  const double A_cboa = p[15];
  const double phi_cboa = p[16];
  const double A_cbophi = p[17];
  const double phi_cbophi = p[18];

  const double N_loss = 1 - K_loss * cumuLoss(t);

  // parameter 31 encodes which tracker frequency model to use
  // right now (0-1) means original model
  // (1-2) means Elba model
  // ... these parameters are currently in the order I added things
  const unsigned int trackerModelNum = p[31];
  if (trackerModelNum == 0) {
    w_cbo = cbo_model_original(w_cbo, t, p + 19);
  } else if (trackerModelNum == 1) {
    w_cbo = cbo_model_elba(w_cbo, t, p + 19);
  } else {
    std::cerr << "Invalid trackerModelNum!" << std::endl;
  }

  // double omega_CBO parameters on the n term
  const double tau_2cbo = p[24];
  const double A_2cbo = p[25];
  const double phi_2cbo = p[26];

  // parameter 32 encodes whether we're fitting in "use field index"
  // mode. if so, the changing CBO frequency is mapped to an n-value
  // and then used to determing the VW and y frequencies, and then adjusted
  // with "fudge" factors to account for imperfections of continuous quad
  // approximations
  const unsigned int use_field_index_flag = p[32];
  // will be updated if we're in "use field index" mode
  double field_index = 1;
  // vertical waist oscillations
  if (use_field_index_flag == 1) {
    // in this case, p[13] is the VW "fudge factor"
    // in units of percent
    const double delta_vw = p[13];
    field_index = n_of_omega_CBO(w_cbo);
    w_vw = (1 + delta_vw / 100) * omega_c * (1 - 2 * sqrt(field_index));
  }

  const double N_vw = 1 + exp(-t / tau_vw) * (A_vw * cos(w_vw * t - phi_vw));

  // vertical betatron oscillations
  const double tau_y = p[27];
  const double A_y = p[28];
  const double phi_y = p[29];

  double w_y = p[30];
  if (A_y != 0 && use_field_index_flag) {
    // now p[30] is the y "fudge factor"
    // same definition as for vw
    const double delta_y = p[30];
    w_y = (1 + delta_y / 100) * omega_c * sqrt(field_index);
  }

  const double N_y =
      A_y != 0 ? 1 + exp(-t / tau_y) * (A_y * cos(w_y * t - phi_y)) : 1;

  // assymetry/phase modulation
  A = A * (1 + cboEnvelope(t, tau_cbo) * A_cboa * cos(w_cbo * t - phi_cboa));
  phi = phi + cboEnvelope(t, tau_cbo) * A_cbophi * cos(w_cbo * t - phi_cbophi);

  const double N_cbo =
      1 + cboEnvelope(t, tau_cbo) * (A_cbo * cos(w_cbo * t - phi_cbo));

  const double N_2cbo =
      1 + exp(-t / tau_2cbo) * (A_2cbo * cos(2 * w_cbo * t - phi_2cbo));

  const double N = N_0 * N_cbo * N_2cbo * N_loss * N_vw * N_y;

  const double wa = wa_ref * (1 + R * 1e-6);

  return N * exp(-t / tau) * (1 + A * cos(wa * t - phi));
}

void createFullFitTF1(const char* tf1Name) {
  new TF1(tf1Name, full_wiggle_fit, 0, 700, n_full_fit_parameters);
}

//
// Some functions to facilitate using a Root::Math::Minimizer
//

class FullFitFunction {
 public:
  FullFitFunction()
      : histToFit(nullptr), tStart(0), tEnd(0), likelihoodMode(false) {}

  double chi2Function(const double* p) const {
    const unsigned int startBin = getStartBin();
    const unsigned int endBin = getEndBin();

    double val = 0;

    for (unsigned int i = startBin; i <= endBin; ++i) {
      double t[1] = {histToFit->GetBinCenter(i)};
      double funcVal = full_wiggle_fit(t, p);
      double histVal = histToFit->GetBinContent(i);

      if (!likelihoodMode) {
        double histVar = pow(histToFit->GetBinError(i), 2);
        // chi2
        val += pow(histVal - funcVal, 2) / histVar;
      } else {
        // negative log likelihood
        if (histVal >= 10.0) {
          // Stirling's approximation
          val -= histVal * (log(funcVal / histVal) + 1) - funcVal;
        } else {
          val -= histVal * log(funcVal) - funcVal - lgamma(histVal + 1);
        }
      }
    }

    if (likelihoodMode) {
      // multiply by 2 if using log likelhood mode
      // so that the minimizer will return the correct errors
      val *= 2;
    }
    return val;
  }

  void setFitStart(double tStartIn) { tStart = tStartIn; }
  void setFitEnd(double tEndIn) { tEnd = tEndIn; }
  void setHistToFit(TH1D* histToFitIn) { histToFit = histToFitIn; }
  void setHistToFit(char* histName) {
    histToFit = (TH1D*)gROOT->FindObject(histName);
  }
  void setLikelihoodMode(bool likelihoodModeIn) {
    likelihoodMode = likelihoodModeIn;
  }

  unsigned int getNIncludedBins() const {
    return getEndBin() - getStartBin() + 1;
  }

 private:
  TH1D* histToFit;
  double tStart;
  double tEnd;
  bool likelihoodMode;

  unsigned int getStartBin() const {
    unsigned int startBin = histToFit->FindBin(tStart);

    if (histToFit->GetBinCenter(startBin) < tStart) {
      startBin += 1;
    }

    return startBin;
  }

  unsigned int getEndBin() const {
    unsigned int endBin = histToFit->FindBin(tEnd);

    if (histToFit->GetBinCenter(endBin) > tEnd) {
      endBin -= 1;
    }

    return endBin;
  }
};

FullFitFunction* fitFunc = nullptr;
ROOT::Math::Functor* functor = nullptr;
ROOT::Math::Minimizer* minimizer = nullptr;
void buildMinimizer(const char* minimizerType = "Minuit2",
                    const char* algoType = "Migrad") {
  if (fitFunc == nullptr) {
    fitFunc = new FullFitFunction;
    functor = new ROOT::Math::Functor(fitFunc, &FullFitFunction::chi2Function,
                                      n_full_fit_parameters);
    minimizer = ROOT::Math::Factory::CreateMinimizer(minimizerType, algoType);
    minimizer->SetFunction(*functor);
  }
}
