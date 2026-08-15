/**
 * @file GKIntegrator.hpp
 * @author Mark Krumholz
 * @brief Adaptive Gauss-Kronrod integration of vector-valued integrands
 * @date 2026-08-15
 */

#ifndef GKINTEGRATOR_HPP
#define GKINTEGRATOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace utils
{

    /**
     * @brief Selects which Gauss-Kronrod pair GKIntegrator uses
     * @details
     * Each enumerator names a (2n+1)-point Kronrod rule paired with
     * its embedded n-point Gauss rule, exactly as the GSL's own
     * gsl_integration_qagNN family does (e.g. GK15 is the pairing
     * gsl_integration_qk15 uses) -- see GKRule's own comment for where
     * each pair's actual abscissae/weights come from. A higher-order
     * pair uses more points per subinterval, converging in fewer
     * adaptive subdivisions for a smooth integrand, at the cost of
     * more integrand evaluations (and more of F's own shared work
     * repeated) per subdivision actually taken.
     */
    enum class GKOrder : std::uint8_t
    {
        GK15, /**< 15-point Kronrod rule, embedding a 7-point Gauss rule */ // NOLINT(readability-identifier-naming) -- named after the GSL's own gsl_integration_qk15, not a generic enumerator this codebase's usual camelBack convention fits
        GK21, /**< 21-point Kronrod rule, embedding a 10-point Gauss rule */ // NOLINT(readability-identifier-naming) -- see GK15
        GK31, /**< 31-point Kronrod rule, embedding a 15-point Gauss rule */ // NOLINT(readability-identifier-naming) -- see GK15
        GK41, /**< 41-point Kronrod rule, embedding a 20-point Gauss rule */ // NOLINT(readability-identifier-naming) -- see GK15
        GK51, /**< 51-point Kronrod rule, embedding a 25-point Gauss rule */ // NOLINT(readability-identifier-naming) -- see GK15
        GK61  /**< 61-point Kronrod rule, embedding a 30-point Gauss rule */ // NOLINT(readability-identifier-naming) -- see GK15
    };

    /**
     * @brief Abscissae and weights for a single Gauss-Kronrod pair
     * @tparam Order Which pair -- see GKOrder's own comment
     * @details
     * Only ever used via one of the six explicit specializations below
     * (one per GKOrder enumerator); the primary template is
     * deliberately left undefined, so instantiating GKRule<Order> for
     * anything else is a compile error rather than silently
     * ill-defined. Every specialization's own ngk/ng/xgk/wg/wgk
     * values are copied verbatim from the GSL's own integration/
     * qkNN.c (NN = 15, 21, 31, 41, 51, 61), which in turn credits them
     * to L. W. Fullerton, Bell Labs, Nov. 1981, evaluated with 80
     * decimal digit arithmetic -- reproduced here (as std::array
     * rather than the GSL's own flat C arrays, but otherwise
     * unchanged) purely so GKIntegrator does not need to link against
     * the GSL's own integration routines to get at them. Only the
     * first half (plus,
     * for an odd point count, the shared middle point at x = 0) of
     * each symmetric rule's own abscissae/weights are stored, exactly
     * as the GSL's own arrays do: xgk[ngk - 1] is always the shared
     * middle abscissa (x = 0), xgk[1], xgk[3], ... are the Gauss
     * rule's own (positive) abscissae, and xgk[0], xgk[2], ... are
     * the additional abscissae the Kronrod extension adds -- see
     * qk15.c's own comment for the identical convention this mirrors.
     */
    template <GKOrder Order>
    struct GKRule;

    /** @brief GKRule specialization for GKOrder::GK15 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK15>
    {
        static constexpr std::size_t ng = 4;   /**< Number of stored Gauss rule weights */
        static constexpr std::size_t ngk = 8;  /**< Number of stored Kronrod rule abscissae/weights */

        /** @brief Abscissae of the 15-point Kronrod rule -- see GKRule's own comment */
        static constexpr std::array<double, ngk> xgk = {
            0.991455371120812639206854697526329,
            0.949107912342758524526189684047851,
            0.864864423359769072789712788640926,
            0.741531185599394439863864773280788,
            0.586087235467691130294144838258730,
            0.405845151377397166906606412076961,
            0.207784955007898467600689403773245,
            0.000000000000000000000000000000000
        };

        /** @brief Weights of the 7-point Gauss rule -- see GKRule's own comment */
        static constexpr std::array<double, ng> wg = {
            0.129484966168869693270611432679082,
            0.279705391489276667901467771423780,
            0.381830050505118944950369775488975,
            0.417959183673469387755102040816327
        };

        /** @brief Weights of the 15-point Kronrod rule -- see GKRule's own comment */
        static constexpr std::array<double, ngk> wgk = {
            0.022935322010529224963732008058970,
            0.063092092629978553290700663189204,
            0.104790010322250183839876322541518,
            0.140653259715525918745189590510238,
            0.169004726639267902826583426598550,
            0.190350578064785409913256402421014,
            0.204432940075298892414161999234649,
            0.209482141084727828012999174891714
        };
    };

    /** @brief GKRule specialization for GKOrder::GK21 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK21>
    {
        static constexpr std::size_t ng = 5;
        static constexpr std::size_t ngk = 11;

        static constexpr std::array<double, ngk> xgk = {
            0.995657163025808080735527280689003,
            0.973906528517171720077964012084452,
            0.930157491355708226001207180059508,
            0.865063366688984510732096688423493,
            0.780817726586416897063717578345042,
            0.679409568299024406234327365114874,
            0.562757134668604683339000099272694,
            0.433395394129247190799265943165784, // NOLINT(modernize-use-std-numbers) -- a Kronrod abscissa, not std::numbers::log10e, despite the numerical coincidence
            0.294392862701460198131126603103866,
            0.148874338981631210884826001129720,
            0.000000000000000000000000000000000
        };

        static constexpr std::array<double, ng> wg = {
            0.066671344308688137593568809893332,
            0.149451349150580593145776339657697,
            0.219086362515982043995534934228163,
            0.269266719309996355091226921569469,
            0.295524224714752870173892994651338
        };

        static constexpr std::array<double, ngk> wgk = {
            0.011694638867371874278064396062192,
            0.032558162307964727478818972459390,
            0.054755896574351996031381300244580,
            0.075039674810919952767043140916190,
            0.093125454583697605535065465083366,
            0.109387158802297641899210590325805,
            0.123491976262065851077958109831074,
            0.134709217311473325928054001771707,
            0.142775938577060080797094273138717,
            0.147739104901338491374841515972068,
            0.149445554002916905664936468389821
        };
    };

    /** @brief GKRule specialization for GKOrder::GK31 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK31>
    {
        static constexpr std::size_t ng = 8;
        static constexpr std::size_t ngk = 16;

        static constexpr std::array<double, ngk> xgk = {
            0.998002298693397060285172840152271,
            0.987992518020485428489565718586613,
            0.967739075679139134257347978784337,
            0.937273392400705904307758947710209,
            0.897264532344081900882509656454496,
            0.848206583410427216200648320774217,
            0.790418501442465932967649294817947,
            0.724417731360170047416186054613938,
            0.650996741297416970533735895313275,
            0.570972172608538847537226737253911,
            0.485081863640239680693655740232351,
            0.394151347077563369897207370981045,
            0.299180007153168812166780024266389,
            0.201194093997434522300628303394596,
            0.101142066918717499027074231447392,
            0.000000000000000000000000000000000
        };

        static constexpr std::array<double, ng> wg = {
            0.030753241996117268354628393577204,
            0.070366047488108124709267416450667,
            0.107159220467171935011869546685869,
            0.139570677926154314447804794511028,
            0.166269205816993933553200860481209,
            0.186161000015562211026800561866423,
            0.198431485327111576456118326443839,
            0.202578241925561272880620199967519
        };

        static constexpr std::array<double, ngk> wgk = {
            0.005377479872923348987792051430128,
            0.015007947329316122538374763075807,
            0.025460847326715320186874001019653,
            0.035346360791375846222037948478360,
            0.044589751324764876608227299373280,
            0.053481524690928087265343147239430,
            0.062009567800670640285139230960803,
            0.069854121318728258709520077099147,
            0.076849680757720378894432777482659,
            0.083080502823133021038289247286104,
            0.088564443056211770647275443693774,
            0.093126598170825321225486872747346,
            0.096642726983623678505179907627589,
            0.099173598721791959332393173484603,
            0.100769845523875595044946662617570,
            0.101330007014791549017374792767493
        };
    };

    /** @brief GKRule specialization for GKOrder::GK41 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK41>
    {
        static constexpr std::size_t ng = 10;
        static constexpr std::size_t ngk = 21;

        static constexpr std::array<double, ngk> xgk = {
            0.998859031588277663838315576545863,
            0.993128599185094924786122388471320,
            0.981507877450250259193342994720217,
            0.963971927277913791267666131197277,
            0.940822633831754753519982722212443,
            0.912234428251325905867752441203298,
            0.878276811252281976077442995113078,
            0.839116971822218823394529061701521,
            0.795041428837551198350638833272788,
            0.746331906460150792614305070355642,
            0.693237656334751384805490711845932, // NOLINT(modernize-use-std-numbers) -- a Kronrod abscissa, not std::numbers::ln2, despite the numerical coincidence
            0.636053680726515025452836696226286,
            0.575140446819710315342946036586425,
            0.510867001950827098004364050955251,
            0.443593175238725103199992213492640,
            0.373706088715419560672548177024927,
            0.301627868114913004320555356858592,
            0.227785851141645078080496195368575,
            0.152605465240922675505220241022678,
            0.076526521133497333754640409398838,
            0.000000000000000000000000000000000
        };

        static constexpr std::array<double, ng> wg = {
            0.017614007139152118311861962351853,
            0.040601429800386941331039952274932,
            0.062672048334109063569506535187042,
            0.083276741576704748724758143222046,
            0.101930119817240435036750135480350,
            0.118194531961518417312377377711382,
            0.131688638449176626898494499748163,
            0.142096109318382051329298325067165,
            0.149172986472603746787828737001969,
            0.152753387130725850698084331955098
        };

        static constexpr std::array<double, ngk> wgk = {
            0.003073583718520531501218293246031,
            0.008600269855642942198661787950102,
            0.014626169256971252983787960308868,
            0.020388373461266523598010231432755,
            0.025882133604951158834505067096153,
            0.031287306777032798958543119323801,
            0.036600169758200798030557240707211,
            0.041668873327973686263788305936895,
            0.046434821867497674720231880926108,
            0.050944573923728691932707670050345,
            0.055195105348285994744832372419777,
            0.059111400880639572374967220648594,
            0.062653237554781168025870122174255,
            0.065834597133618422111563556969398,
            0.068648672928521619345623411885368,
            0.071054423553444068305790361723210,
            0.073030690332786667495189417658913,
            0.074582875400499188986581418362488,
            0.075704497684556674659542775376617,
            0.076377867672080736705502835038061,
            0.076600711917999656445049901530102
        };
    };

    /** @brief GKRule specialization for GKOrder::GK51 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK51>
    {
        static constexpr std::size_t ng = 13;
        static constexpr std::size_t ngk = 26;

        static constexpr std::array<double, ngk> xgk = {
            0.999262104992609834193457486540341,
            0.995556969790498097908784946893902,
            0.988035794534077247637331014577406,
            0.976663921459517511498315386479594,
            0.961614986425842512418130033660167,
            0.942974571228974339414011169658471,
            0.920747115281701561746346084546331,
            0.894991997878275368851042006782805,
            0.865847065293275595448996969588340,
            0.833442628760834001421021108693570,
            0.797873797998500059410410904994307,
            0.759259263037357630577282865204361,
            0.717766406813084388186654079773298,
            0.673566368473468364485120633247622,
            0.626810099010317412788122681624518,
            0.577662930241222967723689841612654, // NOLINT(modernize-use-std-numbers) -- a Kronrod abscissa, not std::numbers::inv_sqrt3, despite the numerical coincidence
            0.526325284334719182599623778158010,
            0.473002731445714960522182115009192,
            0.417885382193037748851814394594572,
            0.361172305809387837735821730127641,
            0.303089538931107830167478909980339,
            0.243866883720988432045190362797452,
            0.183718939421048892015969888759528,
            0.122864692610710396387359818808037,
            0.061544483005685078886546392366797,
            0.000000000000000000000000000000000
        };

        static constexpr std::array<double, ng> wg = {
            0.011393798501026287947902964113235,
            0.026354986615032137261901815295299,
            0.040939156701306312655623487711646,
            0.054904695975835191925936891540473,
            0.068038333812356917207187185656708,
            0.080140700335001018013234959669111,
            0.091028261982963649811497220702892,
            0.100535949067050644202206890392686,
            0.108519624474263653116093957050117,
            0.114858259145711648339325545869556,
            0.119455763535784772228178126512901,
            0.122242442990310041688959518945852,
            0.123176053726715451203902873079050
        };

        static constexpr std::array<double, ngk> wgk = {
            0.001987383892330315926507851882843,
            0.005561932135356713758040236901066,
            0.009473973386174151607207710523655,
            0.013236229195571674813656405846976,
            0.016847817709128298231516667536336,
            0.020435371145882835456568292235939,
            0.024009945606953216220092489164881,
            0.027475317587851737802948455517811,
            0.030792300167387488891109020215229,
            0.034002130274329337836748795229551,
            0.037116271483415543560330625367620,
            0.040083825504032382074839284467076,
            0.042872845020170049476895792439495,
            0.045502913049921788909870584752660,
            0.047982537138836713906392255756915,
            0.050277679080715671963325259433440,
            0.052362885806407475864366712137873,
            0.054251129888545490144543370459876,
            0.055950811220412317308240686382747,
            0.057437116361567832853582693939506,
            0.058689680022394207961974175856788,
            0.059720340324174059979099291932562,
            0.060539455376045862945360267517565,
            0.061128509717053048305859030416293,
            0.061471189871425316661544131965264,
            0.061580818067832935078759824240066
        };
    };

    /** @brief GKRule specialization for GKOrder::GK61 -- see GKRule's own comment */
    template <>
    struct GKRule<GKOrder::GK61>
    {
        static constexpr std::size_t ng = 15;
        static constexpr std::size_t ngk = 31;

        static constexpr std::array<double, ngk> xgk = {
            0.999484410050490637571325895705811,
            0.996893484074649540271630050918695,
            0.991630996870404594858628366109486,
            0.983668123279747209970032581605663,
            0.973116322501126268374693868423707,
            0.960021864968307512216871025581798,
            0.944374444748559979415831324037439,
            0.926200047429274325879324277080474,
            0.905573307699907798546522558925958,
            0.882560535792052681543116462530226,
            0.857205233546061098958658510658944,
            0.829565762382768397442898119732502,
            0.799727835821839083013668942322683,
            0.767777432104826194917977340974503,
            0.733790062453226804726171131369528,
            0.697850494793315796932292388026640,
            0.660061064126626961370053668149271,
            0.620526182989242861140477556431189,
            0.579345235826361691756024932172540,
            0.536624148142019899264169793311073,
            0.492480467861778574993693061207709,
            0.447033769538089176780609900322854,
            0.400401254830394392535476211542661,
            0.352704725530878113471037207089374,
            0.304073202273625077372677107199257,
            0.254636926167889846439805129817805,
            0.204525116682309891438957671002025,
            0.153869913608583546963794672743256,
            0.102806937966737030147096751318001,
            0.051471842555317695833025213166723,
            0.000000000000000000000000000000000
        };

        static constexpr std::array<double, ng> wg = {
            0.007968192496166605615465883474674,
            0.018466468311090959142302131912047,
            0.028784707883323369349719179611292,
            0.038799192569627049596801936446348,
            0.048402672830594052902938140422808,
            0.057493156217619066481721689402056,
            0.065974229882180495128128515115962,
            0.073755974737705206268243850022191,
            0.080755895229420215354694938460530,
            0.086899787201082979802387530715126,
            0.092122522237786128717632707087619,
            0.096368737174644259639468626351810,
            0.099593420586795267062780282103569,
            0.101762389748405504596428952168554,
            0.102852652893558840341285636705415
        };

        static constexpr std::array<double, ngk> wgk = {
            0.001389013698677007624551591226760,
            0.003890461127099884051267201844516,
            0.006630703915931292173319826369750,
            0.009273279659517763428441146892024,
            0.011823015253496341742232898853251,
            0.014369729507045804812451432443580,
            0.016920889189053272627572289420322,
            0.019414141193942381173408951050128,
            0.021828035821609192297167485738339,
            0.024191162078080601365686370725232,
            0.026509954882333101610601709335075,
            0.028754048765041292843978785354334,
            0.030907257562387762472884252943092,
            0.032981447057483726031814191016854,
            0.034979338028060024137499670731468,
            0.036882364651821229223911065617136,
            0.038678945624727592950348651532281,
            0.040374538951535959111995279752468,
            0.041969810215164246147147541285970,
            0.043452539701356069316831728117073,
            0.044814800133162663192355551616723,
            0.046059238271006988116271735559374,
            0.047185546569299153945261478181099,
            0.048185861757087129140779492298305,
            0.049055434555029778887528165367238,
            0.049795683427074206357811569379942,
            0.050405921402782346840893085653585,
            0.050881795898749606492297473049805,
            0.051221547849258772170656282604944,
            0.051426128537459025933862879215781,
            0.051494729429451567558340433647099
        };
    };

    /**
     * @brief The outcome of applying a single Gauss-Kronrod rule to one interval
     * @details
     * What GKIntegrator::quadSingle() returns, and what
     * GKIntegrator::integrate() bisects and re-integrates as it
     * adaptively refines the overall integration domain -- see both
     * of their own comments.
     */
    struct GKIntegrationInterval
    {
        double a_;                  /**< Lower limit of this interval */
        double b_;                  /**< Upper limit of this interval */
        std::vector<double> quad_;  /**< The Kronrod rule's own quadrature estimate over [a_, b_], one value per quantity f_ returns */
        std::vector<double> err_;   /**< This interval's own error estimate, one value per quantity f_ returns -- see quadSingle()'s own comment for how */
    };

    /**
     * @class GKIntegrator
     * @brief Adaptively integrate a vector-valued function of one variable
     * @tparam F Type of the callable integrand: must be invocable as
     *   f(x, ...), where x is the (scalar) integration variable and
     *   ... is any number of additional arguments forwarded from
     *   integrate(); must return a contiguous container of nInt
     *   doubles (e.g. std::vector<double> or std::array<double, nInt>)
     *   -- the value of the integrand at x.
     * @tparam Order Which Gauss-Kronrod pair to integrate each
     *   subinterval with -- see GKOrder's own comment; selected at
     *   compile time (rather than a runtime option) so the
     *   corresponding GKRule<Order> specialization's own fixed-size
     *   xgk/wg/wgk arrays are available with no indirection or
     *   runtime dispatch.
     * @details
     * An adaptation of the GSL's gsl_integration_qag() to vector-valued
     * integrands, using C++ templating rather than gsl_function's
     * pointer-to-void mechanism to pass any extra arguments f itself
     * requires beyond the integration variable.
     */
    template <class F, GKOrder Order>
    class GKIntegrator
    {
    public:

        /**
         * @brief Construct a GKIntegrator
         * @param f The integrand; see this class's own @tparam F
         * @param nInt Number of quantities f returns per point
         * @param maxIter Maximum number of integrate()'s own bisection
         *   iterations (0 = unlimited) -- see its own comment; not a
         *   count of raw evaluations of f_ itself
         * @param absTol Required absolute error
         * @param relTol Required relative error
         */
        GKIntegrator(
            F f,
            std::size_t nInt,
            std::size_t maxIter = 0,
            double absTol = 0.0,
            double relTol = 1e-6
        ) :
            f_(std::move(f)),
            nInt_(nInt),
            maxIter_(maxIter),
            absTol_(absTol),
            relTol_(relTol)
        { }

        /**
         * @brief Apply a single, non-adaptive Gauss-Kronrod rule to one interval
         * @param a Lower limit of the interval
         * @param b Upper limit of the interval
         * @param args Any additional arguments f_ requires, forwarded
         *   to it unchanged after x at every point evaluated
         * @return A GKIntegrationInterval with a_/b_ set to a/b, quad_
         *   set to the ngk-point Kronrod rule's own estimate of the
         *   integral (one value per quantity f_ returns), and err_ set
         *   to an error estimate for each of those same values -- the
         *   absolute difference between the Kronrod estimate and the
         *   embedded (lower-order) Gauss rule's own estimate of the
         *   same integral
         * @details
         * A direct translation of slug2's own
         * slug_imf_integrator<T>::integrate_gk (src/utils/
         * slug_imf_integrator.cpp), generalized from a single
         * scalar-or-vector quantity T to an arbitrary, runtime-sized
         * nInt_ of them, and with the IMF weighting integrate_gk
         * itself folds in (imf_val) dropped entirely -- f_ is
         * evaluated on its own here, unweighted; PDFIntegrator is
         * where that weighting belongs instead, layered on top of this
         * class, not inside it.
         *
         * Builds the symmetric grid of gknum = 2 * ngk - 1 abscissae
         * spanning [a, b] from GKRule<Order>::xgk (see its own
         * comment for the half-plus-center storage convention this
         * mirrors), evaluates f_ at every one of them, then forms two
         * weighted sums over those same nInt_-length values: the full
         * Kronrod sum (all gknum points, weighted by
         * GKRule<Order>::wgk) and the embedded Gauss sum (only the
         * subset of points the lower-order Gauss rule itself uses,
         * weighted by GKRule<Order>::wg) -- including the shared
         * center point (x = (a + b) / 2) in the Gauss sum too,
         * whenever GKRule<Order>::ngk is even (see GKRule's own
         * comment for why that parity is exactly the right test).
         * Both sums are scaled by the interval's own half-length to
         * convert from the standard [-1, 1] quadrature domain to
         * [a, b], and their absolute difference is returned as this
         * call's own error estimate -- a cheap, standard proxy for the
         * Kronrod estimate's own true error, since the embedded Gauss
         * rule is enough lower-order that the two estimates should
         * differ substantially only where the Kronrod one is itself
         * unreliable.
         */
        template <class... Args>
        [[nodiscard]] auto quadSingle(const double a, const double b, Args&&... args) const
        -> GKIntegrationInterval
        {
            using Rule = GKRule<Order>;
            constexpr std::size_t ngk = Rule::ngk;
            constexpr std::size_t gknum = (2 * ngk) - 1;

            const double center = 0.5 * (a + b);
            const double halfLength = 0.5 * (b - a);

            std::array<double, gknum> xk{};
            for (std::size_t i = 0; i < gknum / 2; ++i)
            {
                xk[i] = center - (halfLength * Rule::xgk[i]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < gknum / 2 == ngk - 1 < ngk == Rule::xgk.size() by construction
                xk[gknum - i - 1] = center + (halfLength * Rule::xgk[i]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- gknum - i - 1 >= gknum - gknum/2 > 0 and < gknum, and i < ngk as above
            }
            xk[gknum / 2] = center; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- gknum / 2 < gknum by construction

            // Evaluate f_ at every point first, rather than
            // interleaving evaluation with the summation loops below,
            // to mirror integrate_gk's own structure -- each point's
            // own nInt_ values are copied out of whatever contiguous
            // container f_ returns (see this class's own @tparam F)
            // into a plain std::vector<double>, since nInt_ itself is
            // only known at runtime. args is forwarded exactly once,
            // into this tuple, rather than at each of the gknum calls
            // below: repeatedly forwarding the same forwarding-reference
            // parameter would risk moving from an rvalue-bound argument
            // on its first use, leaving it in a moved-from state for
            // every later point -- std::apply instead hands every call
            // the tuple's own lvalue elements.
            const std::tuple<Args...> argsTuple(std::forward<Args>(args)...);
            std::vector<std::vector<double>> fVal(gknum);
            for (std::size_t i = 0; i < gknum; ++i)
            {
                const auto val = std::apply(
                    [this, &xk, i](const auto&... unpackedArgs) { return std::invoke(f_, xk[i], unpackedArgs...); }, // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < gknum == xk.size() by construction
                    argsTuple);
                fVal[i].assign(std::begin(val), std::end(val));
            }

            std::vector<double> result(nInt_, 0.0);
            std::vector<double> gaussQuad(nInt_, 0.0);

            // Central point: always part of the Kronrod sum (weight
            // wgk[ngk - 1], the last stored weight -- see GKRule's own
            // comment for why); part of the Gauss sum too only when
            // ngk is even (see this function's own comment for why).
            const std::size_t centerIdx = gknum / 2;
            for (std::size_t k = 0; k < nInt_; ++k)
            {
                result[k] = fVal[centerIdx][k] * Rule::wgk[ngk - 1]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ngk - 1 < ngk == Rule::wgk.size() by construction
            }
            if constexpr (ngk % 2 == 0)
            {
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    gaussQuad[k] = fVal[centerIdx][k] * Rule::wg[(ngk / 2) - 1]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- ngk / 2 - 1 < ng == Rule::wg.size() whenever ngk is even, by the Gauss/Kronrod pairing GKRule's own comment describes
                }
            }

            // Points shared between the Gauss and Kronrod sums
            for (std::size_t i = 0; i < (ngk - 1) / 2; ++i)
            {
                const std::size_t p1 = (2 * i) + 1;
                const std::size_t p2 = gknum - (2 * i) - 2;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    const double sum = fVal[p1][k] + fVal[p2][k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1, p2 < gknum == fVal.size() by construction
                    gaussQuad[k] += Rule::wg[i] * sum; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < (ngk - 1) / 2 <= ng == Rule::wg.size(), by the Gauss/Kronrod pairing GKRule's own comment describes
                    result[k] += Rule::wgk[p1] * sum; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1 < ngk == Rule::wgk.size() by construction
                }
            }

            // Points that appear only in the Kronrod sum
            for (std::size_t i = 0; i < ngk / 2; ++i)
            {
                const std::size_t p1 = 2 * i;
                const std::size_t p2 = gknum - (2 * i) - 1;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    result[k] += Rule::wgk[p1] * (fVal[p1][k] + fVal[p2][k]); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p1 < ngk == Rule::wgk.size(), and p1, p2 < gknum == fVal.size(), by construction
                }
            }

            std::vector<double> err(nInt_);
            for (std::size_t k = 0; k < nInt_; ++k)
            {
                result[k] *= halfLength;
                gaussQuad[k] *= halfLength;
                err[k] = std::abs(result[k] - gaussQuad[k]);
            }
            return { a, b, std::move(result), std::move(err) };
        }

        /**
         * @brief Adaptively integrate f_ over [a, b]
         * @param a Lower limit of integration
         * @param b Upper limit of integration
         * @param args Any additional arguments f_ requires; see
         *   quadSingle()'s own args parameter
         * @return The integral estimate, one value per quantity f_
         *   returns (length nInt_)
         * @details
         * Starts from a single quadSingle() call over the whole
         * [a, b] interval; if that single interval's own largest
         * elementwise error (max_k |err_[k]|) is already below absTol_,
         * or its own largest elementwise relative error
         * (max_k |err_[k]| / |quad_[k]|) is already below relTol_,
         * returns immediately -- no bisection needed.
         *
         * Otherwise, repeatedly bisects whichever interval currently
         * contributes the largest relative error to the running total
         * (quadSum, tracked incrementally: each iteration undoes one
         * interval's own old contribution to quadSum/errSum, replaces
         * that interval in place with a fresh quadSingle() over its
         * own left half, appends a new interval for its own right
         * half, and adds both halves' own contributions back in),
         * stopping -- and returning quadSum -- as soon as either the
         * largest element of errSum drops below absTol_, or the
         * largest elementwise relative error (|errSum[k]| / |quadSum[k]|)
         * drops below relTol_, or the number of bisection iterations
         * exceeds maxIter_ (0 = unlimited) -- not a count of raw
         * evaluations of f_ itself, which each iteration costs two
         * further quadSingle() calls' worth of (2 * (2 * ngk - 1)
         * further evaluations).
         *
         * Which interval to bisect next is decided by each interval's
         * own relative error against the *global* running total
         * (|intervals[i].err_[k]| / |quadSum[k]|, maximized over k),
         * not against that interval's own local quad_ -- so bisection
         * effort concentrates on whichever interval's own uncertainty
         * matters most to the overall answer, mirroring slug2's own
         * identical convention in slug_imf_integrator<T>::
         * integrate_range (see quadSingle()'s own comment for that
         * function's sibling, integrate_gk, which this class's
         * quadSingle() itself translates).
         */
        template <class... Args>
        [[nodiscard]] auto integrate(const double a, const double b, Args&&... args) const -> std::vector<double>
        {
            // args is forwarded exactly once, into this tuple, for the
            // same reason quadSingle() itself does this -- see its own
            // comment: integrate() calls quadSingle() repeatedly below,
            // and each of those calls must see the same, still-valid
            // arguments, not ones already moved from on a previous call.
            const std::tuple<Args...> argsTuple(std::forward<Args>(args)...);
            const auto callQuadSingle = [this, &argsTuple](const double lo, const double hi) -> GKIntegrationInterval
            {
                return std::apply(
                    [this, lo, hi](const auto&... unpackedArgs) { return quadSingle(lo, hi, unpackedArgs...); },
                    argsTuple);
            };

            std::vector<GKIntegrationInterval> intervals;
            intervals.push_back(callQuadSingle(a, b));

            const double absErr = *std::ranges::max_element(intervals[0].err_);
            if (absErr < absTol_) { return intervals[0].quad_; }
            {
                double maxRelErr = 0.0;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    maxRelErr = std::max(maxRelErr,
                        std::abs(intervals[0].err_[k] / intervals[0].quad_[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == intervals[0].err_.size() == intervals[0].quad_.size() by construction
                }
                if (maxRelErr < relTol_) { return intervals[0].quad_; }
            }

            std::vector<double> quadSum = intervals[0].quad_;
            std::vector<double> errSum = intervals[0].err_;
            std::size_t itCounter = 1;
            std::size_t intervalPtr = 0;

            while (true)
            {
                const double xMid = 0.5 * (intervals[intervalPtr].a_ + intervals[intervalPtr].b_);
                const double bOld = intervals[intervalPtr].b_; // intervals[intervalPtr] itself is about to be overwritten below, taking its own b_ with it -- the right half bisected off it still needs the original upper limit, not the new one
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    quadSum[k] -= intervals[intervalPtr].quad_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == quadSum.size() == intervals[intervalPtr].quad_.size() by construction
                    errSum[k] -= intervals[intervalPtr].err_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for errSum/err_
                }

                intervals[intervalPtr] = callQuadSingle(intervals[intervalPtr].a_, xMid);
                intervals.push_back(callQuadSingle(xMid, bOld));

                const auto& left = intervals[intervalPtr];
                const auto& right = intervals.back();
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    quadSum[k] += left.quad_[k] + right.quad_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == quadSum.size() == left.quad_.size() == right.quad_.size() by construction
                    errSum[k] += left.err_[k] + right.err_[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for errSum/err_
                }

                ++itCounter;
                if (maxIter_ != 0 && itCounter > maxIter_) { return quadSum; }

                double maxAbsErr = 0.0;
                double maxRelErr = 0.0;
                for (std::size_t k = 0; k < nInt_; ++k)
                {
                    maxAbsErr = std::max(maxAbsErr, std::abs(errSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == errSum.size() by construction
                    maxRelErr = std::max(maxRelErr, std::abs(errSum[k] / quadSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above, for quadSum
                }
                if (maxAbsErr < absTol_ || maxRelErr < relTol_) { return quadSum; }

                double bestRelErr = -1.0;
                std::size_t best = 0;
                for (std::size_t i = 0; i < intervals.size(); ++i)
                {
                    double relErr = 0.0;
                    for (std::size_t k = 0; k < nInt_; ++k)
                    {
                        relErr = std::max(relErr,
                            std::abs(intervals[i].err_[k] / quadSum[k])); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < nInt_ == intervals[i].err_.size() == quadSum.size() by construction
                    }
                    if (relErr > bestRelErr) { bestRelErr = relErr; best = i; }
                }
                intervalPtr = best;
            }
        }

    private:

        F f_;                  /**< The integrand */
        std::size_t nInt_;     /**< Number of quantities f returns per point */
        std::size_t maxIter_;  /**< Maximum number of integrate()'s own bisection iterations */
        double absTol_;        /**< Required absolute error */
        double relTol_;        /**< Required relative error */
    };

} // namespace utils

#endif // GKINTEGRATOR_HPP
