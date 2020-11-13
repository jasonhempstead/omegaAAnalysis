// functions needed for fitting the precession histograms
//
// Aaron Fienberg
// September 2018

// Updating for kernel method error usage
// Jason Hempstead
// November 2020

#include <iostream>

#include "TH1.h"
#include "TF1.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"

#include <cstdlib>

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

//see https://muon.npl.washington.edu/elog/g2/Tracking+Analysis/205
double cbo_model_run2(double w_cbo, double t, const double* p) {
  const double cbo_exp_a = p[0];
  const double tau_a = p[1];
    
  static constexpr double cutoff_t = 1;
  if (t >= cutoff_t) {
    return w_cbo + cbo_exp_a * exp(-t / tau_a) / t;
  } else {
    return cbo_model_run2(w_cbo, cutoff_t, p);
  }

}

//see https://gm2-docdb.fnal.gov/cgi-bin/private/RetrieveFile?docid=24197
double kicker_eddy_current(double wa, double phi, double t, const double* p){
    const double eddy_tau = p[1];
    const double eddy_Delta = p[0]*1e-9; // convert to ppb
    
    return phi - wa*eddy_tau*eddy_Delta* (1 - exp(-t / eddy_tau) );
}

// approximate is ok ;)
constexpr double omega_c = 42.1153248017936;

// get field-index from CBO frequency
double n_of_omega_CBO(double omega_cbo) {
  return (omega_cbo * (2 * omega_c - omega_cbo)) / pow(omega_c, 2);
}

// constexpr unsigned int n_full_fit_parameters = 27;
constexpr unsigned int n_full_fit_parameters = 36;
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
  const double tau_vw = p[10]; //won't be used, but we can still import it, i guess
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
    
  //Nov 2020, unclear what we'll need as far as tracker model moving forward
  const unsigned int trackerModelNum = p[31];
  if (trackerModelNum == 0) {
    w_cbo = cbo_model_original(w_cbo, t, p + 19);
  } else if (trackerModelNum == 1) {
    w_cbo = cbo_model_elba(w_cbo, t, p + 19); //currently in use, double exponential; chag
  } else if (trackerModelNum == 2) {
    w_cbo = cbo_model_run2(w_cbo, t, p + 19); //going to steal back some parameters for moments-based fitting
  } else {
    std::cerr << "Invalid trackerModelNum!" << std::endl;
  }

  // double omega_CBO parameters on the n term
  //const double tau_2cbo = p[24]; won't need with moments-based
  const double A_2cbo = p[23];
  const double phi_2cbo = p[24];
  const double A_cbo2 = p[21]; // be careful, this might fail depending on tracker model; will fix later
  const double phi_cbo2 = p[22]; // same as above

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
    
  const unsigned int kickerModelNum = p[33];
  if (kickerModelNum == 0) {
      //do nothing for now
  } else if(kickerModelNum == 1){
      //alter accumulated phase and omega_a based on eddy current perturbation of field
      
      phi = kicker_eddy_current(wa_ref, phi, t, p+34);//need to change this function a bit
  } else {
      std::cerr << "Invalid kicker eddy current model number!!" << std::endl;
  }

  //const double N_vw = 1 + exp(-t / tau_vw) * (A_vw * cos(w_vw * t - phi_vw));

  // vertical betatron oscillations
  const double tau_y = p[25];
  const double A_y = p[26];
  const double phi_y = p[27];

  double w_y = p[28];
    
  const double A_y2 = p[29];
  const double phi_y2 = p[30];

  // same warnings as above, be careful with the different tracker models; i recommend "2" while testing the moments-based fitting
    
  if (A_y != 0 && use_field_index_flag) {
    // now p[30] is the y "fudge factor"
    // same definition as for vw
    const double delta_y = p[30];
    w_y = (1 + delta_y / 100) * omega_c * sqrt(field_index);
  }

  //const double N_y =
  //    A_y != 0 ? 1 + exp(-t / tau_y) * (A_y * cos(w_y * t - phi_y)) : 1;

  // assymetry/phase modulation
  A = A * (1 + A_cboa * exp(-t / tau_cbo) * cos(w_cbo * t - phi_cboa)); //removed CBO envelope
  phi = phi + A_cbophi * exp(-t / tau_cbo) * cos(w_cbo * t - phi_cbophi); //removed CBO envelope

  /*const double N_cbo =
      1 + cboEnvelope(t, tau_cbo) * (A_cbo * cos(w_cbo * t - phi_cbo));

  const double N_2cbo =
      1 + exp(-t / tau_2cbo) * (A_2cbo * cos(2 * w_cbo * t - phi_2cbo));*/
    
  const double N_x =
      1 + exp(-t / tau_cbo) * (A_cbo * cos(w_cbo * t - phi_cbo) )
        + exp(-2*t / tau_cbo) * ( A_cbo2 * cos(w_cbo * t - phi_cbo2) +
                                 A_2cbo * cos(2 * w_cbo * t - phi_2cbo)); //also removed cboEnvelope
    
  const double N_y =
      1 + exp(-t / tau_y) * (A_y * cos(w_y * t - phi_y))
        + exp(-2*t / tau_y) * A_y2 * cos(w_y * t - phi_y2) +
        + exp(-t / tau_vw) * A_vw * cos(w_vw*t - phi_vw); //removed envelope

  const double N = N_0 * N_x * N_loss * N_y;
    //Former N_vw and N_y combined to one term (N_y)
    //N_cbo and N_2cbo combined to one term (N_x)

  const double wa = wa_ref * (1 + R * 1e-6);

  return N * exp(-t / tau) * (1 + A * cos(wa * t - phi));
}

void createFullFitTF1(const char* tf1Name) {
  new TF1(tf1Name, full_wiggle_fit, 0, 700, n_full_fit_parameters);
}

constexpr unsigned int n_five_fit_param = 15;
double fiveparam_wiggle_fit(const double* x, const double* p) {
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

  const double N_loss = 1 - K_loss * cumuLoss(t);
  const double N_vw = 1 + exp(-t / tau_vw) * (A_vw * cos(w_vw * t - phi_vw));

  const double N_cbo =
      1 + cboEnvelope(t, tau_cbo) * (A_cbo * cos(w_cbo * t - phi_cbo));

    const double N = N_0 * N_cbo * N_loss * N_vw;

  const double wa = wa_ref * (1 + R * 1e-6);

  return N * exp(-t / tau) * (1 + A * cos(wa * t - phi));
}

void createFiveParamFitTF1(const char* tf1Name) {
  new TF1(tf1Name, fiveparam_wiggle_fit, 0, 700, n_five_fit_param);
}

//
// Some functions to facilitate using a Root::Math::Minimizer
//


// look into grabbing an array of values to form the correlation matrix
// a la the PU uncertainty enhancement
class FullFitFunction {
 public:
  FullFitFunction()
      : histToFit(nullptr), tStart(0), tEnd(0), likelihoodMode(false), kernelMode(false) {}

  double chi2Function(const double* p) const {
    const unsigned int startBin = getStartBin();
    const unsigned int endBin = getEndBin();

    double val = 0;

    if(!kernelMode){
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
    }else{
        vector<double> histMinusFunc(endBin - startBin + 1);
    //std::cout<<"starting to populate chi2 value"<<std::endl;
        for(unsigned int j = 0; j < endBin - startBin + 1; j++){

            double t1[1] = {histToFit->GetBinCenter(j+startBin)};
            double funcVal1 = full_wiggle_fit(t1, p);
            double histVal1 = histToFit->GetBinContent(j+startBin);
            histMinusFunc[j] = histVal1 - funcVal1;

        }
        for(unsigned int j = 0; j < endBin - startBin + 1; j++){
            for(unsigned int k = 0; k < endBin - startBin + 1; k++){

                // chi2
                val += histMinusFunc[j] * this->covInverse[j][k] * histMinusFunc[k];

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
    
    void setKernelMethod(bool kernelMethodIn) {
      kernelMode = kernelMethodIn;
        if(kernelMode){
            this->covInverse.clear();
            this->covInverse = getCovInverse(getStartBin(), getEndBin());
            std::cout<<"one entry of inverse cov : " << this->covInverse[0][0]<<std::endl;
        }else{
            this->covInverse.clear();
        }
    }

  unsigned int getNIncludedBins() const {
    return getEndBin() - getStartBin() + 1;
  }

  std::vector<std::vector<long double>> covInverse;
    
 private:
  TH1D* histToFit;
  double tStart;
  double tEnd;
  bool likelihoodMode;
  bool kernelMode;

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

  vector<vector<long double>> getCovInverse(unsigned int startBin, unsigned int endBin){

          /*VectorXd contents(endBin - startBin + 1);
          VectorXd nValues(endBin - startBin + 1);
          MatrixXd adjustmentToGetN(endBin - startBin + 1, endBin - startBin + 1);
          MatrixXd covMatrix(endBin - startBin + 1, endBin - startBin + 1);*/

      std::vector<long double> cVals(endBin - startBin + 1, 0.0);
      std::vector<long double> bVals(endBin - startBin + 1, 0.0);
      std::vector<long double> aVals(endBin - startBin + 1, 0.0);
      std::cout << "size of vector is "<< endBin - startBin + 1 << std::endl;
      aVals[0] = 0.0;
      /*std::vector<long double> zVals(endBin - startBin + 1, 0.0);
      std::vector<long double> yVals(endBin - startBin + 1, 0.0);
      std::vector<long double> phiVals(endBin - startBin + 1, 0.0);*/

      std::vector<std::vector<long double>> altCovMatrix(endBin - startBin + 1);

      std::vector<long double> bValsN(endBin - startBin + 1, 3.0/4.0);
      std::vector<long double> aValsN(endBin - startBin + 1, 1.0/8.0);
      aValsN[0] = 0.0;
      std::vector<long double> cValsN(endBin - startBin + 1, 1.0/8.0);
      cValsN[endBin - startBin] = 0.0;
      std::vector<long double> nVals(endBin - startBin + 1, 0.0);
      std::vector<long double> conN(endBin - startBin + 1, 0.0);

      for(unsigned int j = 0; j < endBin - startBin + 1; j++){
          //contents(j) = max(histToFit->GetBinContent(j+startBin),0.0);
          //nValues(j) = max(histToFit->GetBinContent(j+startBin),0.0);
          conN[j] = max(histToFit->GetBinContent(j+startBin),0.0);
          /*for(unsigned int k = 0; k<endBin - startBin + 1; k++){
              if(k - j == 1 || j - k == 1){
                  adjustmentToGetN(j,k) = 1.0/8.0;
              }else if(j == k){
                  adjustmentToGetN(j,k) = 3.0/4.0;
              }else{
                  adjustmentToGetN(j,k) = 0.0;
              }
          }*/
      }


      /*std::cout<<"getting N values" <<std::endl;
      nValues = adjustmentToGetN.ldlt().solve(contents);
      std::cout<<"N values by solver : " << nValues <<std::endl;*/

      //doing some Ax = b nonsense, solving for x (N)
      nVals = triDiagSolver(aValsN, bValsN, cValsN, conN);
      std::cout << "first element of n is " << nVals[0] << std::endl;
      conN.clear();
      cValsN.clear();
      bValsN.clear();
      aValsN.clear();

      for(unsigned int j = 0; j < endBin - startBin + 1; j++){
          for(unsigned int k = 0; k < endBin - startBin + 1; k++){
              if(k - j == 1 || j - k == 1){
                  //covMatrix(j,k) = 1.0/12.0 * (nValues(j)+nValues(k));
                  if(j>k){
                      aVals[j] = 1.0/12.0 * (nVals[j]+nVals[k]);
                      cVals[j-1] = aVals[j];
                  }
              }else if(j == k){
                  //covMatrix(j,k) = (7.0/12.0) * nValues(k);
                  bVals[j] = (7.0/12.0) * nVals[k];
                  if(k != endBin-startBin+1){
                      //covMatrix(j,k)+=(1.0/24.0)*nValues(k+1);
                      bVals[j]+=(1.0/24.0)*nVals[k+1];

                  }
                  if(k != 0){
                      //covMatrix(j,k)+=(1.0/24.0)*nValues(k-1);
                      bVals[j]+=(1.0/24.0)*nVals[k-1];

                  }
              }else{
                  //covMatrix(j,k) = 0.0;
              }
          }
      }

      vector<long double> identityCol(endBin - startBin + 1, 0.0);
      identityCol[0] = 1.0;
      for(unsigned int col = 0; col < endBin - startBin + 1; col++){
          if(col > 0){
              identityCol[col - 1] = 0.0;
              identityCol[col] = 1.0;
          }
          altCovMatrix.at(col) = triDiagSolver(aVals, bVals, cVals, identityCol);
      }

      /*
      zVals[0] = bVals[0];
      std::cout<<"z val at 0 is : "<<zVals[0]<<std::endl;
      std::cout<<"z val at 1 is : "<<zVals[1]<<std::endl;
      for(unsigned int i = 1; i < zVals.size(); i++){
          if(i == 1){
              zVals[i] = bVals[i]*zVals[i-1] - pow(aVals[i],2.0)*1.0;
          }else{
              zVals[i] = bVals[i]*zVals[i-1] - pow(aVals[i],2.0)*zVals[i-2];
              std::cout << "first term : " << bVals[i]*zVals[i-1] << std::endl;
              std::cout << "second term : " << pow(aVals[i],2.0)*zVals[i-2] << std::endl;
          }
          if(zVals[i] == 0){
              std::cout << "z val is 0!" << std::endl;
          }else if(isnan(zVals[i])){
              std::cout << "z val is nan! at " << i << std::endl;
              std::cout << "a val is : " << aVals[i] << " and bVal is : " << bVals[i]<<std::endl;
              std::cout << "prev z values are : " << zVals[i-1] << " and " << zVals[i-2] << std::endl;
          }else{
              std::cout << "z val is " << zVals[i] << std::endl;
              std::cout << "prev z val is " << zVals[i - 1] << std::endl;
              std::cout << "a val is : " << aVals[i] << " and bVal is : " << bVals[i-1]<<std::endl;
              if(i > 1){
                  std::cout << "other prev z is : " << zVals[i - 2] << std::endl;
              }
          }
      }
      yVals[yVals.size() - 1] = bVals[bVals.size() - 1];
      for(int j = yVals.size() - 2; j >= 0; j--){
          if(j == yVals.size() - 2){
              yVals[j] = bVals[j]*yVals[j+1] - pow(aVals[j+1],2.0)*1.0;
          }else{
              yVals[j] = bVals[j]*yVals[j+1] - pow(aVals[j+1],2.0)*yVals[j+2];
          }
          if(yVals[j] == 0){
              std::cout << "y val is 0!" << std::endl;
          }else if(isnan(yVals[j])){
              std::cout << "y val is nan! at " << j << std::endl;
          }
      }
      for(unsigned int j = 0; j < phiVals.size(); j++){
          double tempDenom = bVals[j];
          //tempDenom goes nan probalby when z or y is 0
          if(j == 1){
              if(zVals[j-1] == 0){
                  tempDenom -= pow(aVals[j],2)/zVals[j-1];
              }
          }else if(j > 1){
              if(zVals[j-1] != 0){
                  tempDenom -= pow(aVals[j],2)*zVals[j-2]/zVals[j-1];
              }
          }
          if(j == phiVals.size() - 2){
              if(yVals[j+1] != 0){
                  tempDenom -= pow(aVals[j+1],2)/yVals[j+1];
              }
          }else if(j < phiVals.size() - 2){
              if(yVals[j+1] != 0){
                  tempDenom -= pow(aVals[j+1],2)*yVals[j+2]/yVals[j+1];
              }
          }
          phiVals[j] = 1.0 / tempDenom;
          //std::cout<<"phi value at " << j << " is " << phiVals[j] << std::endl;
      }
      for(int j = 0; j < phiVals.size(); j++){//upper triangle; j right, i down
          altCovMatrix[j].resize(endBin - startBin + 1);
          double multValue = 1.0;
          double zyLow = 1.0;
          double zyHigh = 1.0;
          for(int i = j-1; i >= 0; i--){
              multValue *= aVals[i+1];
              if(i > 0){
                  zyHigh = zVals[i-1];
              }else if(i == 0){
                  zyHigh = 1.0;
              }
              if(j > 0){
                  zyLow = zVals[j-1];
              }else if( j== 0){
                  zyLow = 1.0;
              }
              if(multValue == 0){
                  zyLow = 1.0;
              }
              if(zyLow == 0){
                  zyLow = 1.0;
                  std::cout<<"okay adjusting a bit..."<<std::endl;
                  std::cout<<"phi val is : " << phiVals[j] << std::endl;
                  std::cout<<"mult value is : "<<multValue << std::endl;
                  std::cout<<"zyHigh value is : "<<zyHigh << std::endl;
              }
              altCovMatrix[j][i] = pow(-1,j-i)*phiVals[j]*multValue*zyHigh/zyLow;
          }
          altCovMatrix[j][j] = phiVals[j];
          multValue = 1.0;
          zyLow = 1.0;
          zyHigh = 1.0;
          for(unsigned int i = j+1; i < phiVals.size(); i ++){//lower triangle
              multValue*= aVals[i];
              if(i < phiVals.size() - 1){
                  zyHigh = yVals[i+1];
              }else if(i == phiVals.size() - 1){
                  zyHigh = 1.0;
              }
              if(j < phiVals.size() - 1){
                  zyLow = yVals[j+1];
              }else if(j < phiVals.size() == 1){
                  zyLow = 1.0;
              }
              if(multValue == 0){
                  zyLow = 1.0;
              }
              if(zyLow == 0){
                  zyLow = 1.0;
                  std::cout<<"okay adjusting a bit..."<<std::endl;
                  std::cout<<"phi val is : " << phiVals[j] << std::endl;
                  std::cout<<"mult value is : "<<multValue << std::endl;
                  std::cout<<"zyHigh value is : "<<zyHigh << std::endl;
              }
              altCovMatrix[j][i] = pow(-1,j-i)*phiVals[j]*multValue*zyHigh/zyLow;
          }
      }*/


      //std::cout<<"Here is cov: "<<std::endl <<covMatrix<<std::endl;
      //Eigen::FullPivLU<MatrixXd> lu(covMatrix);
      //Eigen::LLT<MatrixXd> lltOfCov(covMatrix);
      //std::cout<<"is this invertible? "<< lu.isInvertible() << std::endl;
      //std::cout<<"taking the inverse"<<std::endl;
      //return covMatrix.llt().solve(MatrixXd::Identity(endBin - startBin + 1, endBin - startBin + 1));
      //MatrixXd covInverseMat = covMatrix.llt().solve(MatrixXd::Identity(endBin - startBin + 1, endBin - startBin + 1));
      /*std::cout<<"Here is inverse by eigen : "<<std::endl <<covInverseMat<<std::endl;*/
      //std::cout<<"Here is the alternate cov by tridiagonal: "<<std::endl <<altCovMatrix<<std::endl;

      for(int k = 0; k < altCovMatrix.size(); k++){
          for(int j = 0; j < altCovMatrix.size(); j++){
              if(isnan(altCovMatrix[k][j])){
                  std::cout<<"ah shoot, not this again..."<<std::endl;
              }
          }
      }
      std::cout<<"one entry of returned matrix is : " <<altCovMatrix[0][0]<<std::endl;
      std::cout<<"last entry of returned matrix is : " <<altCovMatrix[altCovMatrix.size() - 1][altCovMatrix.size() - 1]<<std::endl;
      return altCovMatrix;
  }

  vector<long double> triDiagSolver(vector<long double> a, vector<long double> b, vector<long double> c, vector<long double> d){

      if(a.size() != b.size() || b.size() != c.size() || c.size() != d.size()){
          std::cout << "how do you expect to solve this thing!" << std::endl;
          return {};
      }

      vector<long double> aCopy(a);
      vector<long double> bCopy(b);
      vector<long double> cCopy(c);
      vector<long double> dCopy(d);

      vector<long double> x;

      x.resize(dCopy.size());

      double w = 1.0;
      for(unsigned int i = 1; i < dCopy.size(); i++){
          w = aCopy[i]/bCopy[i-1];
          bCopy[i] -= w*cCopy[i-1];
          dCopy[i] -= w*dCopy[i-1];
      }
      x[x.size()-1] = dCopy[x.size() - 1] / bCopy[x.size() - 1];
      for(int j = x.size()-2; j>=0; j--){
          x[j] = (dCopy[j] - cCopy[j]*x[j+1])/bCopy[j];
      }

      return x;
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
