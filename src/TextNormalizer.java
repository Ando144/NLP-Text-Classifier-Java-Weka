import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleStreamFilter;

import java.util.regex.Pattern;

/**
 * TextNormalizer — Weka SimpleStreamFilter
 *
 * Mezu elektronikoen testu gordina normalizatzen du StringToWordVector-en
 * aurretik aplikatzeko. Helburu nagusia: spam-ek erabiltzen dituen
 * evasion teknikak neutralizatzea eta zarata kentzea.
 *
 * NORMALIZAZIO URRATSAK (hurrenkeran aplikatzen dira):
 *
 *   1. URL normalizazioa:
 *      "http://nvhomevalues.com/?partid=s23" → "URLTOKEN"
 *      Arrazoibidea: URL zehatza ez da informatiboa klasifikatzeko;
 *      URLaren PRESENTZIAK bai (spam seinale garrantzitsua).
 *
 *   2. Email helbide normalizazioa:
 *      "raymon@hhdx.com" → "EMAILTOKEN"
 *      Arrazoibidea: email helbide zehatza ez da informatiboa;
 *      baina presentziak spam seinale gisa funtziona dezake.
 *
 *   3. Letra suelten kolapso:
 *      "v . i . c . o . d . i . n" → "vicodin"
 *      "c i a l i s" → "cialis"
 *      Arrazoibidea: spam-ek filtro-ihes gisa erabiltzen duen teknika.
 *      Regex-ak letra bakar bat + zuriune/puntu errepikatua detektatzen du.
 *
 *   4. HTML artefaktuen garbiketa:
 *      "bgcolor=8080ff", "cellpadding=5", "trtdtable" → ""
 *      Arrazoibidea: HTML kodea parseatu gabe sartzen da batzuetan;
 *      token hauek ez dute informazio semantikorik.
 *
 *   5. Karaktere errepikatuen murrizketa:
 *      "!!!!!!!" → "!", "......." → "."
 *      Arrazoibidea: spam-ek arreta erakartzeko erabiltzen du.
 *      Presentzia mantendu baina zarata murriztu.
 *
 *   6. Espazio anitzen normalizazioa:
 *      "hitz    bat" → "hitz bat"
 *      Arrazoibidea: aurreko pausoen ondorioz ager daitezkeen zuriune
 *      anitzak garbitzea.
 *
 * ERABILERA EmailVectorizerAndSelector.java-n:
 *   // StringToWordVector AURRETIK deitu:
 *   TextNormalizer normalizer = new TextNormalizer();
 *   normalizer.setInputFormat(train);
 *   Instances trainNorm = Filter.useFilter(train, normalizer);
 *   Instances devNorm   = Filter.useFilter(dev,   normalizer);
 *   Instances testNorm  = Filter.useFilter(test,  normalizer);
 *   // Ondoren StringToWordVector trainNorm-ekin erabili.
 *
 * KONPILAZIO ADIBIDEA:
 *   javac -cp lib/weka.jar -d bin src/TextNormalizer.java
 *
 * @author WekaProyecto2026 Taldea
 */
public class TextNormalizer extends SimpleStreamFilter {

    private static final long serialVersionUID = 1L;

    // -------------------------------------------------------
    // REGEX PATROIAK — behin konpilatuta, errendimendua hobea
    // -------------------------------------------------------

    /**
     * URL patroia: http/https/ftp protokoloekin hasten diren URLak.
     * Adib: "http : / / nvhomevalues . com / ? partid = s 23"
     * Oharra: Weka-k zuriuneak txertatzen ditu URL-etan (tokenizazioa),
     * beraz bi bertsioak hartzen ditugu kontuan.
     */
    private static final Pattern URL_PATTERN = Pattern.compile(
        "https?\\s*:\\s*/\\s*/\\s*\\S+|ftp\\s*:\\s*/\\s*/\\s*\\S+",
        Pattern.CASE_INSENSITIVE
    );

    /**
     * Email helbide patroia.
     * Adib: "raymon@hhdx.com", "stock65@yahoo.com"
     */
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}"
    );

    /**
     * Letra suelten patroia: letra bakar bat, ondoren zuriune/puntu eta
     * letra bakar bat, hiru aldiz gutxienez.
     * Adib: "v . i . c . o . d . i . n", "c i a l i s", "l r c j"
     *
     * Logika: ([a-z] zatia/zuriunea) 3+ aldiz jarraian.
     */
    private static final Pattern SPACED_LETTERS_PATTERN = Pattern.compile(
        "\\b([a-z])([\\s.]+[a-z]){3,}\\b"
    );

    /**
     * HTML atributu patroia: "izen=balioa" formatuko tokenak.
     * Adib: "bgcolor=8080ff", "cellpadding=5", "width=20"
     */
    private static final Pattern HTML_ATTR_PATTERN = Pattern.compile(
        "\\b[a-z]+=[a-zA-Z0-9#]+\\b",
        Pattern.CASE_INSENSITIVE
    );

    /**
     * HTML tag zatiak: trtd, tdtd, brbr, nbsp eta antzekoak.
     * Adib: "trtdtable", "brbrbrbr", "tdtd"
     */
    private static final Pattern HTML_FRAGMENT_PATTERN = Pattern.compile(
        "\\b(tr|td|br|th|li|ul|ol|div|span|img|href|src|nbsp|bgcolor|cellpad){2,}\\b",
        Pattern.CASE_INSENSITIVE
    );

    /**
     * Karaktere errepikatuen patroia: 3+ aldiz jarraian dagoen edozein
     * ez-alfanumeriko.
     * Adib: "!!!!!!!" → "!", "......." → ".", "-------" → "-"
     */
    private static final Pattern REPEATED_PUNCT_PATTERN = Pattern.compile(
        "([^a-zA-Z0-9\\s])\\1{2,}"
    );

    /**
     * Espazio anitzen patroia: bi zuriune edo gehiago jarraian.
     */
    private static final Pattern MULTI_SPACE_PATTERN = Pattern.compile(
        "\\s{2,}"
    );

    // -------------------------------------------------------
    // SimpleStreamFilter METODO NAGUSIAK
    // -------------------------------------------------------

    @Override
    public String globalInfo() {
        return "Mezu elektronikoen testu gordina normalizatzen du StringToWordVector-en " +
               "aurretik aplikatzeko. URLak, email helbideak, letra suelteak, HTML " +
               "artefaktuak eta karaktere errepikatuak normalizatzen ditu.";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();
        result.enable(Capability.NO_CLASS);
        return result;
    }

    /**
     * Irteera formatua sarrera formatuaren berdina da:
     * string atributua string atributu gisa mantentzen da,
     * baina edukia normalizatuta.
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    /**
     * Instantzia bakarra prozesatzen du: testu atributua aurkitu eta
     * normalizatu, beste atributu guztiak aldatu gabe utziz.
     */
    @Override
    protected Instance process(Instance instance) {
        Instance result = (Instance) instance.copy();

        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).type() == Attribute.STRING
                    && !instance.isMissing(i)) {

                String text = instance.stringValue(i);
                text = normalize(text);

                // Weka-n string balioak indize bidez kudeatzen dira
                result.setValue(i,
                    result.dataset().attribute(i).addStringValue(text));
            }
        }
        return result;
    }

    // -------------------------------------------------------
    // NORMALIZAZIO LOGIKA
    // -------------------------------------------------------

    /**
     * Normalizazio urrats guztiak hurrenkeran aplikatzen ditu.
     * Hurrenkerak garrantzia du: adib. URL-ak kendu aurretik
     * ez da letra suelteak kolapsatu behar (URLetan agertzen baitira).
     *
     * @param text Testu gordina
     * @return Testu normalizatua
     */
    public static String normalize(String text) {
        if (text == null || text.isEmpty()) return text;

        // 1. URL normalizazioa
        text = URL_PATTERN.matcher(text).replaceAll(" URLTOKEN ");

        // 2. Email normalizazioa
        text = EMAIL_PATTERN.matcher(text).replaceAll(" EMAILTOKEN ");

        // 3. Letra suelten kolapso
        //    "v . i . c . o . d . i . n" → "vicodin"
        text = collapseSpacedLetters(text);

        // 4. HTML artefaktuen garbiketa
        text = HTML_ATTR_PATTERN.matcher(text).replaceAll(" ");
        text = HTML_FRAGMENT_PATTERN.matcher(text).replaceAll(" ");

        // 5. Karaktere errepikatuen murrizketa
        text = REPEATED_PUNCT_PATTERN.matcher(text).replaceAll("$1");

        // 6. Espazio anitzen normalizazioa
        text = MULTI_SPACE_PATTERN.matcher(text).replaceAll(" ").trim();

        return text;
    }

    /**
     * Letra suelten sekuentziak kolapsatzen ditu.
     * "v . i . c . o . d . i . n" → "vicodin"
     * "l r c j" → "lrcj"
     *
     * Estrategia: SPACED_LETTERS_PATTERN-ak detektatzen dituen match-ak
     * hartu eta bertatik zuriuneak eta puntuak kendu.
     */
    private static String collapseSpacedLetters(String text) {
        StringBuffer sb = new StringBuffer();
        java.util.regex.Matcher m = SPACED_LETTERS_PATTERN.matcher(text);
        while (m.find()) {
            String matched = m.group();
            // Zuriuneak eta puntuak kendu letra suelteetatik
            String collapsed = matched.replaceAll("[\\s.]+", "");
            m.appendReplacement(sb, collapsed);
        }
        m.appendTail(sb);
        return sb.toString();
    }

    // -------------------------------------------------------
    // MAIN — proba azkarrerako
    // -------------------------------------------------------

    public static void main(String[] args) {
        // Proba txiki bat normalizazioa egiaztatzeko,
        // Weka pipeline osoa erabili gabe.
        String[] probak = {
            "http : / / nvhomevalues . com / ? partid = s 23",
            "v . i . c . o . d . i . n",
            "c i a l i s softabs",
            "l r c j stock symbol",
            "raymon @ hhdx . com",
            "bgcolor=8080ff cellpadding=5 trtdtable brbrbr",
            "buy now !!!!!!!! lowest price .......",
            "mor tg age application accepted"
        };

        System.out.println("=== TextNormalizer proba ===\n");
        for (String p : probak) {
            System.out.println("SARRERA : " + p);
            System.out.println("IRTEERA : " + normalize(p));
            System.out.println();
        }
    }
}
